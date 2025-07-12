import re
from itertools import chain

from transformers import CLIPTokenizer

from app.predefined_styles import fooocus_styles, sai_styles
from app.predefined_styles.schemas import StyleItem

PROMPT_REMOVE_PATTERN = re.compile(r'\s*\{prompt\}[,]?\s*')
CLIP_TOKENIZER_MODEL = 'openai/clip-vit-base-patch32'


class StylesService:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_MODEL)
        self.all_styles = list(chain.from_iterable([fooocus_styles, sai_styles]))

    def truncate_clip_prompt(self, prompt: str, max_tokens: int = 77) -> str:
        clip_tokenizer = self.tokenizer(prompt)
        input_ids = clip_tokenizer.input_ids

        if len(input_ids) > max_tokens:
            input_ids = input_ids[:max_tokens]

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def combined_positive_prompt(self, user_prompt: str, styles: list[StyleItem]):
        if not styles:
            return user_prompt.strip()

        first_style, *rest_styles = styles
        combined_positive = []

        if first_style.positive:
            combined_positive.append(first_style.positive.format(prompt=user_prompt))

        for style in rest_styles:
            if style.positive:
                cleaned = PROMPT_REMOVE_PATTERN.sub(' ', style.positive).strip(' ,')
                if cleaned:
                    combined_positive.append(cleaned)

        combined_positive_unique = list(dict.fromkeys(combined_positive))

        return ', '.join(combined_positive_unique).rstrip(',').strip()

    def combined_negative_prompt(self, styles: list[StyleItem]):
        combined_negative: list[str] = []

        for style in styles:
            if style.negative is not None:
                combined_negative.append(style.negative)

        combined_negative_unique = list(dict.fromkeys(combined_negative))

        return ', '.join(combined_negative_unique).strip()

    def apply_styles(self, user_prompt: str, styles: list[str]):
        selected_styles = [
            style_item for style_item in self.all_styles if style_item.id in styles
        ]

        combined_positive = self.combined_positive_prompt(user_prompt, selected_styles)
        combined_negative = self.combined_negative_prompt(selected_styles)

        truncated_positive_prompt = self.truncate_clip_prompt(combined_positive)
        truncated_negative_prompt = self.truncate_clip_prompt(combined_negative)

        return [truncated_positive_prompt, truncated_negative_prompt]


styles_service = StylesService()
