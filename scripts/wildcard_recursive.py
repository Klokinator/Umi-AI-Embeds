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


def get_index(items, item):
    try:
        return items.index(item)
    except Exception:
        return None


def parse_tag(tag):
    return tag.replace("__", "").replace('<', '').replace('>', '')


def read_file_lines(file):
    f_lines = file.read().splitlines()
    lines = []
    for line in f_lines:
        line = line.strip()
        # Check if we have a line that is not empty or starts with an #
        if line and not line.startswith('#'):
            lines.append(line)
    return lines


class TagLoader:
    files = []
    wildcard_location = os.path.join(pathlib.Path(inspect.getfile(lambda: None)).parent.parent, "wildcards")
    loaded_tags = {}
    missing_tags = set()
    def load_tags(self, filePath):
        filepath_lower = filePath.lower()
        if self.loaded_tags.get(filePath):
            return self.loaded_tags.get(filePath)

        txt_file_path = os.path.join(self.wildcard_location, f'{filePath}.txt')
        yaml_file_path = os.path.join(self.wildcard_location, f'{filePath}.yaml')

        if self.wildcard_location and os.path.isfile(txt_file_path):
            with open(txt_file_path, encoding="utf8") as file:
                self.files.append(f"{filePath}.txt")
                self.loaded_tags[filepath_lower] = read_file_lines(file)
        
        if self.wildcard_location and os.path.isfile(yaml_file_path):
            with open(yaml_file_path, encoding="utf8") as file:
                self.files.append(f"{filePath}.yaml")
                try:
                    data = yaml.safe_load(file)
                    output = {}
                    for item in data:
                       output[item] = {x.lower() for i,x in enumerate(data[item]['Tags'])}
                    self.loaded_tags[filepath_lower] = output
                    print(self.loaded_tags[filepath_lower])
                except yaml.YAMLError as exc:
                    print(exc)

        if not os.path.isfile(yaml_file_path) and not os.path.isfile(txt_file_path):
            self.missing_tags.add(filePath)
        
        return self.loaded_tags.get(filepath_lower) if self.loaded_tags.get(filepath_lower) else []


class TagSelector:
    def __init__(self, tag_loader, options):
        self.tag_loader = tag_loader
        self.previously_selected_tags = {}
        self.selected_options = dict(options).get('selected_options', {})

    def get_tag_choice(self, parsed_tag, tags):
        if self.selected_options.get(parsed_tag.lower()) is not None:
            return tags[self.selected_options.get(parsed_tag.lower())]
        return choices(tags)[0] if len(tags) > 0 else ""

    def get_tag_group_choice(self, parsed_tag, groups, tags):
        print('selected_options', self.selected_options)
        print('groups', groups)
        print('parsed_tag', parsed_tag)
        neg_groups = [x.lower() for x in groups if x.startswith('--')]
        neg_groups_set = {x.replace('--', '') for x in neg_groups}
        any_groups = [{y for i,y in enumerate(x.lower().split('|'))} for x in groups if '|' in x]
        pos_groups = [x.lower() for x in groups if x not in neg_groups and '|' not in x]
        pos_groups_set = {x for x in pos_groups}
        print('pos_groups', pos_groups_set)
        print('negative_groups', neg_groups_set)
        print('any_groups', any_groups)
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
        return choices(candidates)[0] if len(candidates) > 0 else ""


    def select(self, tag, groups=None):
        self.previously_selected_tags.setdefault(tag, 0)

        if self.previously_selected_tags.get(tag) < 100:
            self.previously_selected_tags[tag] += 1
            parsed_tag = parse_tag(tag)
            tags = self.tag_loader.load_tags(parsed_tag)
            if groups and len(groups) > 0:
                return self.get_tag_group_choice(parsed_tag, groups, tags)
            if len(tags) > 0:
                return self.get_tag_choice(parsed_tag, tags)
            return tag
        print(f'loaded tag more than 100 times {tag}')
        return ""


class TagReplacer:
    def __init__(self, tag_selector, options):
        self.tag_selector = tag_selector
        self.options = options
        self.wildcard_regex = re.compile('[_<]_?(.*?)_?[_>]')
        self.opts_regexp = re.compile('(?<=\[)(.*?)(?=\])')

    def replace_wildcard(self, matches):
        if matches is None or len(matches.groups()) == 0:
            return ""

        match = matches.groups()[0]
        match_and_opts = match.split(':')
        if (len(match_and_opts) == 2):
            return self.tag_selector.select(match_and_opts[0], self.opts_regexp.findall(match_and_opts[1]))

        return self.tag_selector.select(match)

    def replace_wildcard_recursive(self, prompt):
        p = self.wildcard_regex.sub(self.replace_wildcard, prompt)
        while p != prompt:
            prompt = p
            p = self.wildcard_regex.sub(self.replace_wildcard, prompt)

        return p

    def replace(self, prompt):
        return self.replace_wildcard_recursive(prompt)


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

    def replace_combinations(self, match):
        if match is None or len(match.groups()) == 0:
            return ""

        variants = [s.strip() for s in match.groups()[0].split("|")]
        weights = [self.get_variant_weight(var) for var in variants]
        variants = [self.get_variant(var) for var in variants]

        summed = sum(weights)
        zero_weights = weights.count(0)
        weights = list(map(lambda x: (100 - summed) / zero_weights if x == 0 else x, weights))
        try:
            picked = choices(variants, weights)[0]
            return picked
        except ValueError as e:
            return ""

    def replace(self, template):
        if template is None:
            return None

        return self.re_combinations.sub(self.replace_combinations, template)


class PromptGenerator:
    def __init__(self, options):
        self.tag_loader = TagLoader()
        self.tag_selector = TagSelector(self.tag_loader, options)
        self.replacers = [TagReplacer(self.tag_selector, options), DynamicPromptReplacer()]

    def use_replacers(self, prompt):
        for replacer in iter(self.replacers):
            prompt = replacer.replace(prompt)

        return prompt

    def generate_single_prompt(self, original_prompt):
        previous_prompt = original_prompt
        prompt = self.use_replacers(original_prompt)
        while previous_prompt != prompt:
            previous_prompt = prompt
            prompt = self.use_replacers(previous_prompt)

        return prompt

    def generate(self, original_prompt, prompt_count):
        return [self.generate_single_prompt(original_prompt) for _ in range(prompt_count)]


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
            parsed_tag = parse_tag(tag);
            location = get_index(self.tag_loader.load_tags(parsed_tag), options[i])
            if location is not None:
                tag_presets[parsed_tag.lower()] = location

        return tag_presets


class PromptGenerator:
    def __init__(self, options):
        self.tag_loader = TagLoader()
        self.tag_selector = TagSelector(self.tag_loader, options)
        self.negative_tag_generator = NegativePromptGenerator()
        self.replacers = [TagReplacer(self.tag_selector, options), DynamicPromptReplacer(), self.negative_tag_generator]

    def use_replacers(self, prompt):
        for replacer in self.replacers:
            prompt = replacer.replace(prompt)

        return prompt

    def generate_single_prompt(self, original_prompt):
        previous_prompt = original_prompt
        prompt = self.use_replacers(original_prompt)
        while previous_prompt != prompt:
            previous_prompt = prompt
            prompt = self.use_replacers(prompt)

        return prompt

    def generate(self, original_prompt, prompt_count):
        return [self.generate_single_prompt(original_prompt) for _ in range(prompt_count)]

    def get_negative_tags(self):
        return self.negative_tag_generator.get_negative_tags()


class NegativePromptGenerator:
    def __init__(self):
        self.re_combinations = re.compile(r"\{([^{}]*)}")
        self.negative_tag = set()

    def strip_negative_tags(self, tags):
        matches = re.findall('\*\*.*?\*\*', tags)
        if matches and len(self.negative_tag) == 0:
            for match in matches:
                self.negative_tag.add(match.replace("**", ""))
                tags = tags.replace(match, "")
        return tags

    def replace(self, prompt):
        return self.strip_negative_tags(prompt)

    def get_negative_tags(self):
        return " ".join(self.negative_tag)


class Script(scripts.Script):
    def title(self):
        return "Prompt generator"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Row():
                enabled = gr.Checkbox(label="UmiAI enabled?", value=True)
                same_seed = gr.Checkbox(label='Use same prompt for each image in a batch?', value=False)
                negative_prompt = gr.Checkbox(label='Allow **negative keywords** from wildcards in Negative Prompts?', value=True)
                shared_seed = gr.Checkbox(label="Always pick the same random/wildcard options with a static seed?", value=False)
            option_generator = OptionGenerator(TagLoader())
            options = [
                gr.Dropdown(label=opt, choices=["RANDOM"] + option_generator.get_option_choices(opt), value="RANDOM")
                for opt in option_generator.get_configurable_options()]

        return [enabled, same_seed, negative_prompt, shared_seed] + options

    def process(self, p, enabled, same_seed, negative_prompt, shared_seed, *args):
        if not enabled:
            return
        TagLoader.files.clear()
        original_prompt = p.all_prompts[0]
        option_generator = OptionGenerator(TagLoader())
        options = {
            'selected_options': option_generator.parse_options(args)
        }
        prompt_generator = PromptGenerator(options)

        for i in range(len(p.all_prompts)):
            if (shared_seed):
                random.seed(p.all_seeds[0 if same_seed else i])
            else:
                random.seed(time.time())

            prompt = p.all_prompts[i]
            prompt = prompt_generator.generate_single_prompt(prompt)
            p.all_prompts[i] = prompt

        if negative_prompt:
            p.negative_prompt = p.negative_prompt + prompt_generator.get_negative_tags()

        if original_prompt != p.all_prompts[0]:
            p.extra_generation_params["Wildcard prompt"] = original_prompt
            p.extra_generation_params["File includes"] = "|".join(TagLoader.files) ## test if it fixes importing
