# -*- coding: utf-8 -*-
"""
CMeKG 关系抽取 (RE) 推理演示 — get_triples() 使用示例

用法:
    python model_re/demo_re.py

依赖:
    - model/medical_re/predicate.json       (谓词定义)
    - model/medical_re/model_re.pkl         (训练好的级联模型权重)
    - model/medical_re/vocab.txt + config.json (BERT 配置)
"""

import json
import sys
import os

# 确保能找到同目录下的 medical_re 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medical_re import load_schema, load_model, get_triples, config


def demo_basic(model4s, model4po):
    """基础示例：药品说明书类文本"""
    print("=" * 60)
    print("示例 1：药品说明书")
    print("=" * 60)
    text = (
        "阿莫西林常用于敏感菌所致的呼吸道感染、泌尿系统感染;"
        "本品为白色或类白色结晶性粉末;"
        "不良反应包括恶心、呕吐、腹泻等胃肠道反应。"
        "对青霉素过敏者禁用。"
    )
    res = get_triples(text, model4s, model4po)
    print(f"输入文本: {text}\n")
    print(f"共 {len(res)} 个分句，抽取结果:")
    for item in res:
        if item["triples"]:
            print(f"  ▸ 「{item['text']}」")
            for s, p, o in item["triples"]:
                print(f"      ({s}, {p}, {o})")
    print()


def demo_empty_input(model4s, model4po):
    """边界情况：空文本"""
    print("=" * 60)
    print("示例 2：空文本")
    print("=" * 60)
    res = get_triples("", model4s, model4po)
    print(f"结果: {res}")
    assert res == [], "空输入应返回空列表"
    print("✓ 空文本处理正确\n")


def demo_short_text(model4s, model4po):
    """短文本：无句号，单句"""
    print("=" * 60)
    print("示例 3：短文本（无句号）")
    print("=" * 60)
    text = "高血压病人不可食用阿莫西林等药物。"
    res = get_triples(text, model4s, model4po)
    print(f"输入文本: {text}")
    print(f"结果: {res}")
    # 注意：无句号时 split('。')[:-1] 为空，结果应为 []
    print(f"(注：文本无句号，get_triples 按句号切分后返回 {len(res)} 句)\n")


def demo_long_text(model4s, model4po):
    """长文本：包含大量重复分句，模拟说明书"""
    print("=" * 60)
    print("示例 4：长文本（多分句）")
    print("=" * 60)
    text = (
        "现病史（1）病史摘要     病人，男，49岁，3小时前解大便后出现右下腹疼痛，右下腹可触及一包块，既往体健。"
        "（2）主诉     右下腹痛并自扪及包块3小时;体格检查体温： T 37.8℃，P 101次／分，呼吸22次/分，BP 100/60mmHg，腹软，未见胃肠型蠕动波，肝脾肋下未及，于右侧腹股沟区可扪及一圆形肿块，约4cm×4cm大小，有压痛、界欠清，且肿块位于腹股沟韧带上内方。"
        "辅助检查（1）实验室检查 血常规：WBC 5.0×109／L，N 78％;尿常规正常。"
        "（2）多普勒超声检查     沿腹股沟纵切可见一多层分布的混合回声区，宽窄不等，远端膨大，边界整齐，长约4～5cm。"
        "（3）腹部X线检查     可见阶梯状液气平。"
    )
    res = get_triples(text, model4s, model4po)
    print(f"输入文本: {text}\n")
    sentence_count = len(res)
    triple_count = sum(len(item["triples"]) for item in res)
    print(f"分句数: {sentence_count}，三元组总数: {triple_count}")
    for item in res:
        if item["triples"]:
            for s, p, o in item["triples"]:
                print(f"  ({s}, {p}, {o})")
    print()


def demo_pretty_json(model4s, model4po):
    """展示 JSON 格式的完整输出"""
    print("=" * 60)
    print("示例 5：结构化 JSON 输出")
    print("=" * 60)
    text = "新冠肺炎患者经常会发热、咳嗽，其病因包括自身免疫系统缺陷。"
    res = get_triples(text, model4s, model4po)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    # 第一步：加载谓词 schema
    schema_path = config.PATH_SCHEMA
    model_path = config.PATH_MODEL

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先下载预训练模型并放置在正确路径。")
        print("参见 CLAUDE.md 中的『模型下载与放置』说明。\n")
        sys.exit(1)

    print(f"加载 schema: {schema_path}")
    print(f"加载模型:   {model_path}")
    print()

    # 第二步：加载 schema 和模型权重
    load_schema(schema_path)
    model4s, model4po = load_model()
    model4s.eval()
    model4po.eval()
    print("模型加载成功\n")

    # 第三步：运行演示
    # demo_basic(model4s, model4po)
    # demo_empty_input(model4s, model4po)
    # demo_short_text(model4s, model4po)
    demo_long_text(model4s, model4po)
    # demo_pretty_json(model4s, model4po)

    print("=" * 60)
    print("所有示例执行完毕。")
