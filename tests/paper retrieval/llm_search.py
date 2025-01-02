import json
import re

from openai import OpenAI  
from tqdm import tqdm


def call_gpt4_api(prompt,model_path):
    client = OpenAI(
        base_url="https://hk.xty.app/v1",
        api_key="", # API密钥
    )
    completion = client.chat.completions.create(
        model=model_path,  # 使用gpt-4模型
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def parse_response(response_text):
    json_start = re.search(r'\s*{', response_text)
    json_end = re.search(r'}\s*', response_text)
    
    if json_start and json_end:
        clean_json_str = response_text[json_start.start():json_end.end()]
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None
    else:
        print("无法找到有效的JSON数据块。")
        return None

def llm_search(survey_abs, paper_abs, top_k, prompt_template, llm_path):
    prompt = prompt_template.format(survey_abs=survey_abs, paper_abs=paper_abs, top_k=top_k)
    result = call_gpt4_api(prompt,llm_path)
    parsed_result = parse_response(result)
    indexes = parsed_results['idx_of_correct_abstract']
    return indexes
    
    