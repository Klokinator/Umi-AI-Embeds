import os
import random
import re
from random import choices


import modules.scripts as scripts
import modules.images as images
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state
from modules.styles import StyleDatabase


def get_index(items, item):
    try:
        return items.index(item)
    except Exception:
        return None


def parse_tag(tag):
    return tag.replace("__", "")


class TagLoader:
    loaded_tags = {}
    missing_tags = set()

    def load_tags(self, filePath):
        filepath_lower = filePath.lower()
        if self.loaded_tags.get(filePath):
            return self.loaded_tags.get(filePath)

        replacement_file = os.path.join(os.getcwd(), f"scripts/wildcards/{filePath}.txt")
        if os.path.exists(replacement_file):
            with open(replacement_file, encoding="utf8") as f:
                lines = f.read().splitlines()
                # remove 'commented out' lines
                self.loaded_tags[filepath_lower] = [item for item in lines if not item.startswith('#')]
        else:
            self.missing_tags.add(filePath)
            return []

        return self.loaded_tags.get(filepath_lower) if self.loaded_tags.get(filepath_lower) else []


class TagSelector:
    previously_selected_tags = {}
    def __init__(self, tag_loader, options):
        self.tag_loader = tag_loader
        self.selected_options = options['selected_options']

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
        self.negative_tag = []

    def strip_negative_tags(self, tags):
        matches = re.findall('\*\*.*?\*\*', tags)
        if matches and len(self.negative_tag) == 0:
            for match in matches:
                self.negative_tag.append(match.replace("**", ""))
                tags = tags.replace(match, "")
        return tags

    def replace(self, prompt):
        return self.strip_negative_tags(prompt)

    def get_negative_tags(self):
        return " ".join(self.negative_tag)

class Script(scripts.Script):
    def title(self):
        return "Prompt generator"

    def ui(self, is_img2img):
        same_seed = gr.Checkbox(label='Use same seed for each image', value=False)
        negative_prompt = gr.Checkbox(label='Generate negative tags?', value=False)
        option_generator = OptionGenerator(TagLoader())
        options = [gr.Dropdown(label=opt, choices=["RANDOM"] + option_generator.get_option_choices(opt), value="RANDOM") for opt in option_generator.get_configurable_options()]

        return [same_seed, negative_prompt] + options

    def run(self, p, same_seed, negative_prompt, *args):
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        option_generator = OptionGenerator(TagLoader())
        options = {}
        options['selected_options'] = option_generator.parse_options(args)

        prompt_generator = PromptGenerator(options)
        all_prompts = prompt_generator.generate(original_prompt, p.batch_size * p.n_iter)
        if negative_prompt:
            p.negative_prompt += prompt_generator.get_negative_tags()

        # TODO: Pregenerate seeds to prevent overlaps when batch_size is > 1
        # Known issue: Clicking "recycle seed" on an image in a batch_size > 1 may not get the correct seed.
        # (unclear if this is an issue with this script or not, but pregenerating would prevent). However,
        # filename and exif data on individual images match correct seeds (testable via sending png info to txt2img).
        all_seeds = []
        infotexts = []

        initial_seed = None
        initial_info = None

        print(f"Will process {p.batch_size * p.n_iter} images in {p.n_iter} batches.")

        state.job_count = p.n_iter
        p.n_iter = 1

        original_do_not_save_grid = p.do_not_save_grid

        p.do_not_save_grid = True

        output_images = []

        for batch_no in range(state.job_count):
            state.job = f"{batch_no+1} out of {state.job_count}"
            # batch_no*p.batch_size:(batch_no+1)*p.batch_size
            p.prompt = all_prompts[batch_no*p.batch_size:(batch_no+1)*p.batch_size]


            if cmd_opts.enable_console_prompts:
                print(f"wildcards.py: {p.prompt}")
            proc = process_images(p)
            output_images += proc.images
            # TODO: Also add wildcard data to exif of individual images, currently only appear on the saved grid.
            infotext = ""

            infotext += "Wildcard prompt: "+original_prompt+"\nExample: "+proc.info
            all_seeds.append(proc.seed)
            infotexts.append(infotext)
            if initial_seed is None:
                initial_info = infotext
                initial_seed = proc.seed
            if not same_seed:
                p.seed = proc.seed

        p.do_not_save_grid = original_do_not_save_grid

        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                infotexts.insert(0, initial_info)
                all_seeds.insert(0, initial_seed)
                output_images.insert(0, grid)

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", all_seeds[0], original_prompt, opts.grid_format, info=initial_info, short_filename=not opts.grid_extended_filename, p=p, grid=True)

        return Processed(p, output_images, initial_seed, initial_info, all_prompts=all_prompts, all_seeds=all_seeds, infotexts=infotexts, index_of_first_image=0)