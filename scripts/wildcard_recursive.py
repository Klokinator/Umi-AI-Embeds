import os
import random
import inspect
import pathlib
import re
import time
import glob
from random import choices
import yaml

import modules.scripts as scripts
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state
from modules import scripts, script_callbacks, shared
from modules.styles import StyleDatabase
import modules.textual_inversion.textual_inversion

from modules.sd_samplers import samplers, samplers_for_img2img


ALL_KEY = 'all yaml files'

def get_index(items, item):
    try:
        return items.index(item)
    except Exception:
        return None


def parse_tag(tag):
    return tag.replace("__", "").replace('<', '').replace('>', '').strip()


def read_file_lines(file):
    f_lines = file.read().splitlines()
    lines = []
    for line in f_lines:
        line = line.strip()
        # Check if we have a line that is not empty or starts with an #
        if line and not line.startswith('#'):
            lines.append(line)
    return lines


# Wildcards
class TagLoader:
    files = []
    wildcard_location = os.path.join(
        pathlib.Path(inspect.getfile(lambda: None)).parent.parent, "wildcards")
    loaded_tags = {}
    missing_tags = set()

    def load_tags(self, file_path, verbose=False, cache_files=True):
        if cache_files and self.loaded_tags.get(file_path):
            return self.loaded_tags.get(file_path)

        txt_file_path = os.path.join(self.wildcard_location, f'{file_path}.txt')
        yaml_file_path = os.path.join(self.wildcard_location,
                                      f'{file_path}.yaml')

        if (file_path == ALL_KEY):
            key = ALL_KEY
        else:
            key = file_path.lower()

        if self.wildcard_location and os.path.isfile(txt_file_path):
            with open(txt_file_path, encoding="utf8") as file:
                self.files.append(f"{file_path}.txt")
                self.loaded_tags[key] = read_file_lines(file)

        if key is ALL_KEY and self.wildcard_location:
            files = glob.glob(os.path.join(self.wildcard_location, '**/*.yaml'), recursive=True)
            output = {}
            for file in files:
                with open(file, encoding="utf8") as file:
                    self.files.append(f"{file_path}.yaml")
                    try:
                        data = yaml.safe_load(file)
                        for item in data:
                            if (hasattr(output, item) and verbose):
                                print(f"Duplicate key {item} in {file}")
                            if data[item] and 'Tags' in data[item]:
                                output[item] = {
                                    x.lower().strip()
                                    for i, x in enumerate(data[item]['Tags'])
                                }
                            else:
                                print(f'Issue with tags found in {file} at item {item}')
                    except yaml.YAMLError as exc:
                        print(exc)
            self.loaded_tags[key] = output

        if self.wildcard_location and os.path.isfile(yaml_file_path):
            with open(yaml_file_path, encoding="utf8") as file:
                self.files.append(f"{file_path}.yaml")
                try:
                    data = yaml.safe_load(file)
                    output = {}
                    for item in data:
                        output[item] = {
                            x.lower().strip()
                            for i, x in enumerate(data[item]['Tags'])
                        }
                    self.loaded_tags[key] = output
                except yaml.YAMLError as exc:
                    print(exc)

        if not os.path.isfile(yaml_file_path) and not os.path.isfile(
                txt_file_path):
            self.missing_tags.add(file_path)

        return self.loaded_tags.get(key) if self.loaded_tags.get(
            key) else []


# <yaml:[tag]> notation
class TagSelector:

    def __init__(self, tag_loader, options):
        self.tag_loader = tag_loader
        self.previously_selected_tags = {}
        self.selected_options = dict(options).get('selected_options', {})
        self.verbose = dict(options).get('verbose', False)
        self.cache_files = dict(options).get('cache_files', True)

    def get_tag_choice(self, parsed_tag, tags):
        if self.selected_options.get(parsed_tag.lower()) is not None:
            return tags[self.selected_options.get(parsed_tag.lower())]
        return choices(tags)[0] if len(tags) > 0 else ""

    def get_tag_group_choice(self, parsed_tag, groups, tags):
        #print('selected_options', self.selected_options)
        #print('groups', groups)
        #print('parsed_tag', parsed_tag)
        neg_groups = [x.strip().lower() for x in groups if x.startswith('--')]
        neg_groups_set = {x.replace('--', '') for x in neg_groups}
        any_groups = [{y.strip()
                       for i, y in enumerate(x.lower().split('|'))}
                      for x in groups if '|' in x]
        pos_groups = [
            x.strip().lower() for x in groups
            if not x.startswith('--') and '|' not in x
        ]
        pos_groups_set = {x for x in pos_groups}
        # print('pos_groups', pos_groups_set)
        # print('negative_groups', neg_groups_set)
        # print('any_groups', any_groups)
        candidates = []
        for tag in tags:
            tag_set = tags[tag]
            if len(list(pos_groups_set & tag_set)) != len(pos_groups_set):
                continue
            if len(list(neg_groups_set & tag_set)) > 0:
                continue
            if len(any_groups) > 0:
                any_groups_found = 0
                for any_group in any_groups:
                    if len(list(any_group & tag_set)) == 0:
                        break
                    any_groups_found += 1
                if len(any_groups) != any_groups_found:
                    continue
            candidates.append(tag)
        if len(candidates) > 0:
            if self.verbose:
                print(
                    f'UmiAI: Found {len(candidates)} candidates for "{parsed_tag}" with tags: {groups}, first 10: {candidates[:10]}'
                )
            return choices(candidates)[0]
        print(
            f'UmiAI: No tag candidates found for: "{parsed_tag}" with tags: {groups}'
        )
        return ""

    def select(self, tag, groups=None):
        self.previously_selected_tags.setdefault(tag, 0)
        if (tag.count(':')==2) or (len(tag) < 2):
            return False
        if self.previously_selected_tags.get(tag) < 50000:
            self.previously_selected_tags[tag] += 1
            parsed_tag = parse_tag(tag)
            tags = self.tag_loader.load_tags(parsed_tag, self.verbose, self.cache_files)
            if groups and len(groups) > 0:
                return self.get_tag_group_choice(parsed_tag, groups, tags)
            if len(tags) > 0:
                return self.get_tag_choice(parsed_tag, tags)
            else:
                print(
                    f'UmiAI: No tags found in wildcard file "{parsed_tag}" or file does not exist'
                )
            return False
        if self.previously_selected_tags.get(tag) == 50000:
            self.previously_selected_tags[tag] += 1
            print(f'Processed more than 50000 tags, this may indicate a tag reference loop. Inspect your tags and remove any loops.')
        return False


class TagReplacer:

    def __init__(self, tag_selector, options):
        self.tag_selector = tag_selector
        self.options = options
        self.wildcard_regex = re.compile('((__|<)(.*?)(__|>))')
        self.opts_regexp = re.compile('(?<=\[)(.*?)(?=\])')

    def replace_wildcard(self, matches):
        if matches is None or len(matches.groups()) == 0:
            return ""

        match = matches.groups()[2]
        match_and_opts = match.split(':')
        if (len(match_and_opts) == 2):
            selected_tags = self.tag_selector.select(
                match_and_opts[0], self.opts_regexp.findall(match_and_opts[1]))
        else:
            global_opts = self.opts_regexp.findall(match)
            if len(global_opts) > 0:
                selected_tags = self.tag_selector.select(ALL_KEY, global_opts)
            else:
                selected_tags = self.tag_selector.select(match)

        if selected_tags:
            return selected_tags
        return matches[0]

    def replace_wildcard_recursive(self, prompt):
        p = self.wildcard_regex.sub(self.replace_wildcard, prompt)
        while p != prompt:
            prompt = p
            p = self.wildcard_regex.sub(self.replace_wildcard, prompt)

        return p

    def replace(self, prompt):
        return self.replace_wildcard_recursive(prompt)


# handle {1$$this | that} notation
class DynamicPromptReplacer:

    def __init__(self):
        self.re_combinations = re.compile(r"\{([^{}]*)}")

    def get_variant_weight(self, variant):
        split_variant = variant.split("%")
        if len(split_variant) == 2:
            num = split_variant[0]
            try:
                return int(num)
            except ValueError:
                print(f'{num} is not a number')
        return 0

    def get_variant(self, variant):
        split_variant = variant.split("%")
        if len(split_variant) == 2:
            return split_variant[1]
        return variant

    def parse_range(self, range_str, num_variants):
        if range_str is None:
            return None

        parts = range_str.split("-")
        if len(parts) == 1:
            low = high = min(int(parts[0]), num_variants)
        elif len(parts) == 2:
            low = int(parts[0]) if parts[0] else 0
            high = min(int(parts[1]),
                       num_variants) if parts[1] else num_variants
        else:
            raise Exception(f"Unexpected range {range_str}")

        return min(low, high), max(low, high)

    def replace_combinations(self, match):
        if match is None or len(match.groups()) == 0:
            return ""

        combinations_str = match.groups()[0]

        variants = [s.strip() for s in combinations_str.split("|")]
        weights = [self.get_variant_weight(var) for var in variants]
        variants = [self.get_variant(var) for var in variants]

        splits = variants[0].split("$$")
        quantity = splits.pop(0) if len(splits) > 1 else str(1)
        variants[0] = splits[0]

        low_range, high_range = self.parse_range(quantity, len(variants))

        quantity = random.randint(low_range, high_range)

        summed = sum(weights)
        zero_weights = weights.count(0)
        weights = list(
            map(lambda x: (100 - summed) / zero_weights
                if x == 0 else x, weights))

        try:
            #print(f"choosing {quantity} tag from:\n{' , '.join(variants)}")
            picked = []
            for x in range(quantity):
                choice = random.choices(variants, weights)[0]
                picked.append(choice)

                index = variants.index(choice)
                variants.pop(index)
                weights.pop(index)

            #print(f"Picked:\n{' , '.join(picked)}\n")
            return " , ".join(picked)
        except ValueError as e:
            return ""

    def replace(self, template):
        if template is None:
            return None

        return self.re_combinations.sub(self.replace_combinations, template)


class OptionGenerator:

    def __init__(self, tag_loader):
        self.tag_loader = tag_loader

    def get_configurable_options(self):
        return self.tag_loader.load_tags('configuration')

    def get_option_choices(self, tag):
        return self.tag_loader.load_tags(parse_tag(tag))

    def parse_options(self, options):
        tag_presets = {}
        for i, tag in enumerate(self.get_configurable_options()):
            parsed_tag = parse_tag(tag)
            location = get_index(self.tag_loader.load_tags(parsed_tag),
                                 options[i])
            if location is not None:
                tag_presets[parsed_tag.lower()] = location

        return tag_presets


class PromptGenerator:

    def __init__(self, options):
        self.tag_loader = TagLoader()
        self.tag_selector = TagSelector(self.tag_loader, options)
        self.negative_tag_generator = NegativePromptGenerator()
        self.settings_generator = SettingsGenerator()
        self.replacers = [
            self.settings_generator,
            TagReplacer(self.tag_selector, options),
            DynamicPromptReplacer()
        ]
        self.verbose = dict(options).get('verbose', False)

    def use_replacers(self, prompt):
        for replacer in self.replacers:
            prompt = replacer.replace(prompt)

        return prompt

    def generate_single_prompt(self, original_prompt):
        previous_prompt = original_prompt
        start = time.time()
        prompt = self.use_replacers(original_prompt)
        while previous_prompt != prompt:
            previous_prompt = prompt
            prompt = self.use_replacers(prompt)
        prompt = self.negative_tag_generator.replace(prompt)
        end = time.time()
        if self.verbose:
            print(f"Prompt generated in {end - start} seconds")

        return prompt

    def get_negative_tags(self):
        return self.negative_tag_generator.get_negative_tags()

    def get_setting_overrides(self):
        return self.settings_generator.get_setting_overrides()


class NegativePromptGenerator:

    def __init__(self):
        self.negative_tag = set()

    def strip_negative_tags(self, tags):
        matches = re.findall('\*\*.*?\*\*', tags)
        if matches:
            for match in matches:
                self.negative_tag.add(match.replace("**", ""))
                tags = tags.replace(match, "")
        return tags

    def replace(self, prompt):
        return self.strip_negative_tags(prompt)

    def get_negative_tags(self):
        return " ".join(self.negative_tag)


# @@settings@@ notation
class SettingsGenerator:

    def __init__(self):
        self.re_setting_tags = re.compile(r"@@(.*?)@@")
        self.setting_overrides = {}
        self.type_mapping = {
            'cfg_scale': float,
            'sampler': str,
            'steps': int,
        }

    def strip_setting_tags(self, prompt):
        matches = self.re_setting_tags.findall(prompt)
        if matches:
            for match in matches:
                for assignment in match.split("|"):
                    key_raw, value = assignment.split("=")
                    if not value:
                        print(
                            f"Invalid setting {assignment}, settings should assign a value"
                        )
                        continue
                    key_found = False
                    for key in self.type_mapping.keys():
                        if key.startswith(key_raw):
                            self.setting_overrides[key] = self.type_mapping[
                                key](value)
                            key_found = True
                            break
                    if not key_found:
                        print(
                            f"Unknown setting {key_raw}, setting should be the starting part of: {', '.join(self.type_mapping.keys())}"
                        )
                prompt = prompt.replace('@@' + match + '@@', "")
        return prompt

    def replace(self, prompt):
        return self.strip_setting_tags(prompt)

    def get_setting_overrides(self):
        return self.setting_overrides


class Script(scripts.Script):
    is_txt2img = False

    embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()

    def title(self):
        return "Prompt generator"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.is_txt2img = is_img2img == False
        with gr.Group():
            with gr.Row():
                enabled = gr.Checkbox(label="UmiAI enabled", value=True)
                verbose = gr.Checkbox(label="Verbose logging", value=False)
                cache_files = gr.Checkbox(label="Cache files", value=True)
                same_seed = gr.Checkbox(label='Same prompt in batch',
                                        value=False)
                negative_prompt = gr.Checkbox(label='**negative keywords**',
                                              value=True)
                shared_seed = gr.Checkbox(label="Static wildcards",
                                          value=False)

            option_generator = OptionGenerator(TagLoader())
            options = [
                gr.Dropdown(label=opt,
                            choices=["RANDOM"] +
                            option_generator.get_option_choices(opt),
                            value="RANDOM")
                for opt in option_generator.get_configurable_options()
            ]

        return [enabled, verbose, cache_files, same_seed, negative_prompt, shared_seed
                ] + options

    def process(self, p, enabled, verbose, cache_files, same_seed, negative_prompt,
                shared_seed, *args):
        if not enabled:
            return

        debug = False

        if debug: print(f'\nModel: {p.sampler_name}, Seed: {int(p.seed)}, Batch Count: {p.n_iter}, Batch Size: {p.batch_size}, CFG: {p.cfg_scale}, Steps: {p.steps}\nOriginal Prompt: "{p.prompt}"\nOriginal Negatives: "{p.negative_prompt}"\n')
        original_prompt = p.all_prompts[0]
        if hasattr(p, "all_negative_prompts"): # hasattr to fix crash on old webui versions
            original_negative = p.all_negative_prompts[0]
        else:
            original_negative = ""

        TagLoader.files.clear()
        original_prompt = p.all_prompts[0]

        option_generator = OptionGenerator(TagLoader())
        options = {
            'selected_options': option_generator.parse_options(args),
            'verbose': verbose,
            'cache_files': cache_files,
        }
        prompt_generator = PromptGenerator(options)

        for cur_count in range(p.n_iter):  #Batch count
            for cur_batch in range(p.batch_size):  #Batch Size

                index = p.batch_size * cur_count + cur_batch

                # pick same wildcard for a given seed
                if (shared_seed):
                    random.seed(p.all_seeds[p.batch_size *cur_count if same_seed else index])
                else:
                    random.seed(time.time()+index*10)
                
                if debug: print(f'{"Batch #"+str(cur_count) if same_seed else "Prompt #"+str(index):=^30}')

                prompt_generator.negative_tag_generator.negative_tag = set()

                prompt = prompt_generator.generate_single_prompt(original_prompt)
                p.all_prompts[index] = prompt

                if debug: print(f'Prompt: "{prompt}"')

                negative = original_negative
                if negative_prompt and hasattr(p, "all_negative_prompts"): # hasattr to fix crash on old webui versions
                    negative += prompt_generator.get_negative_tags()
                    p.all_negative_prompts[index] = negative
                    if debug: print(f'Negative: "{negative}\n"')

                # same prompt per batch
                if (same_seed):
                    for index in range(index, index + p.batch_size):
                        p.all_prompts[index] = prompt
                    break

        def find_sampler_index(sampler_list, value):
            for index, elem in enumerate(sampler_list):
                if elem[0] == value or value in elem[2]:
                    return index

        att_override = prompt_generator.get_setting_overrides()
        #print(att_override)
        for att in att_override.keys():
            if not att.startswith("__"):
                if att == 'sampler':
                    sampler_name = att_override[att]
                    if self.is_txt2img:
                        sampler_index = find_sampler_index(
                            samplers, sampler_name)
                    else:
                        sampler_index = find_sampler_index(
                            samplers_for_img2img, sampler_name)
                    if (sampler_index != None):
                        setattr(p, 'sampler_index', sampler_index)
                    else:
                        print(
                            f"Sampler {sampler_name} not found in prompt {p.all_prompts[0]}"
                        )
                    continue
                setattr(p, att, att_override[att])

        if original_prompt != p.all_prompts[0]:
            p.extra_generation_params["Wildcard prompt"] = original_prompt
            if verbose:
                p.extra_generation_params["File includes"] = "|".join(
                    TagLoader.files)

from modules import sd_hijack
path = os.path.join(scripts.basedir(), "embeddings")
sd_hijack.model_hijack.embedding_db.add_embedding_dir(path)