import os
import random
import inspect
import pathlib
import re
import glob
from random import choices

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
    return tag.replace("__", "")


class TagLoader:
    files = []
    wildcard_location = os.path.join(pathlib.Path(inspect.getfile(lambda: None)).parent, "wildcards")
    loaded_tags = {}
    missing_tags = set()
    print(f"Path is {wildcard_location}")
    def load_tags(self, filePath):
        filepath_lower = filePath.lower()
        if self.loaded_tags.get(filePath):
            return self.loaded_tags.get(filePath)

        file_path = os.path.join(self.wildcard_location, f'{filePath}.txt')

        if self.wildcard_location and os.path.isfile(file_path):
            with open(file_path, encoding="utf8") as f:
                self.files.append(f"{filePath}.txt")
                lines = f.read().splitlines()
                # remove 'commented out' lines
                self.loaded_tags[filepath_lower] = [item for item in lines if not item.startswith('#')]
        else:
            self.missing_tags.add(filePath)
            return []

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

    def select(self, tag):
        self.previously_selected_tags.setdefault(tag, 0)

        if self.previously_selected_tags.get(tag) < 100:
            self.previously_selected_tags[tag] += 1
            parsed_tag = parse_tag(tag)
            tags = self.tag_loader.load_tags(parsed_tag)
            if len(tags) > 0:
                return self.get_tag_choice(parsed_tag, tags)
            return tag
        print(f'loaded tag more than 100 times {tag}')
        return ""


class TagReplacer:
    def __init__(self, tag_selector, options):
        self.tag_selector = tag_selector
        self.options = options
        self.wildcard_regex = re.compile('__(.*?)__')

    def replace_wildcard(self, matches):
        if matches is None or len(matches.groups()) == 0:
            return ""

        match = matches.groups()[0]
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
                same_seed = gr.Checkbox(label='Use same prompt for each image', value=False)
                negative_prompt = gr.Checkbox(label='Generate negative tags?', value=False)
            option_generator = OptionGenerator(TagLoader())
            options = [
                gr.Dropdown(label=opt, choices=["RANDOM"] + option_generator.get_option_choices(opt), value="RANDOM")
                for opt in option_generator.get_configurable_options()]

        return [same_seed, negative_prompt] + options

    def process(self, p, same_seed, negative_prompt, *args):
        TagLoader.files.clear()
        original_prompt = p.all_prompts[0]
        option_generator = OptionGenerator(TagLoader())
        options = {
            'selected_options': option_generator.parse_options(args)
        }
        prompt_generator = PromptGenerator(options)

        print(p.negative_prompt)
        for i in range(len(p.all_prompts)):
            random.seed(p.all_seeds[0 if same_seed else i])
            prompt = p.all_prompts[i]
            prompt = prompt_generator.generate_single_prompt(prompt)
            p.all_prompts[i] = prompt

        if same_seed and negative_prompt:
            p.negative_prompt = prompt_generator.get_negative_tags()
            print('generated negative prompt', p.negative_prompt)

        if original_prompt != p.all_prompts[0]:
            p.extra_generation_params["Wildcard prompt"] = original_prompt
            p.extra_generation_params["File includes"] = "\n".join(TagLoader.files)
