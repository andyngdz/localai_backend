from itertools import chain

from langchain_core.prompts import PromptTemplate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from app.predefined_styles import fooocus_styles, sai_styles
from app.predefined_styles.schemas import StyleItem


class StylesService:
    def __init__(self):
        model_name = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.all_styles = list(chain.from_iterable([fooocus_styles, sai_styles]))
        pass

    def combined_positive_prompt(self, user_prompt: str, styles: list[StyleItem]):
        first_style, *rest_styles = styles
        combined_positive = []

        if first_style.positive:
            combined_positive.append(first_style.positive.format(prompt=user_prompt))

        for style in rest_styles:
            if style.positive:
                cleaned = style.positive.replace('{prompt}', '')
                if cleaned:
                    combined_positive.append(cleaned)

        return ', '.join(combined_positive)

    def combined_negative_prompt(self, styles: list[StyleItem]):
        combined_negative: list[str] = []

        for style in styles:
            if style.negative is not None:
                combined_negative.append(style.negative)

        return ', '.join(combined_negative)

    def apply_styles(self, user_prompt: str, styles: list[str]):
        selected_styles = [
            style_item for style_item in self.all_styles if style_item.id in styles
        ]

        combined_positive = self.combined_positive_prompt(user_prompt, selected_styles)
        combined_negative = self.combined_negative_prompt(selected_styles)

        template_positive_str = 'Optimize this positive prompt: {positive}'
        prompt_positive_template = PromptTemplate(
            input_variables=['positive'],
            template=template_positive_str,
        )
        final_positive_prompt = prompt_positive_template.format(
            positive=combined_positive,
        )

        template_negative_str = 'Optimize this negative prompt: {negative}'
        prompt_negative_template = PromptTemplate(
            input_variables=['negative'],
            template=template_negative_str,
        )
        final_negative_prompt = prompt_negative_template.format(
            negative=combined_negative,
        )

        return [final_positive_prompt, final_negative_prompt]


styles_service = StylesService()
