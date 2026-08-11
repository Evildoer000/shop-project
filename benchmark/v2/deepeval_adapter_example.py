from __future__ import annotations

"""
DeepEval 接入示例。

这个文件故意不加入线上依赖，也不会被普通 benchmark 自动执行。安装 deepeval 后，
把下面的示例移动到 server/tests/evals/ 并接入真实 Orchestrator 输出即可。

项目原则：
1. 价格、类目、商品ID、Recall、nDCG 等确定性指标不交给 LLM Judge。
2. DeepEval 只评价软约束、回答证据忠实性、推荐理由和多轮目标完成度。
3. Judge 使用项目自己的模型适配器，不要求上传数据到外部评估平台。
"""

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_approved_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if not case.get("enabled", True):
                continue
            if case.get("review", {}).get("status") != "approved":
                continue
            cases.append(case)
    return cases


def build_soft_constraint_rubric(case: dict[str, Any]) -> list[str]:
    retrieval = case["expected"]["retrieval"]
    constraints = retrieval.get("soft_constraints") or []
    labels = [str(item.get("label") or item.get("key") or "") for item in constraints]
    return [
        f"判断推荐商品是否有证据支持这些软需求：{'、'.join(labels)}。",
        "商品资料没有提到某属性时必须视为未知，不得脑补为满足。",
        "若商品违反类型、价格等硬约束，本项最高不得给及格分。",
        "评分原因必须引用输入中提供的商品证据。",
    ]


def deterministic_expected_product_ids(case: dict[str, Any], minimum_grade: int = 2) -> set[str]:
    """正式评估只读取人工 approved_grade，不读取机器 proposed_grade。"""
    judged = case["expected"]["retrieval"].get("judged_products") or []
    return {
        str(item["product_id"])
        for item in judged
        if item.get("approved_grade") is not None and int(item["approved_grade"]) >= minimum_grade
    }


def example_deepeval_test() -> None:
    """
    伪代码展示最终 pytest 形态，避免当前项目在未审核数据集时误跑收费 Judge。

    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    @pytest.mark.semantic
    @pytest.mark.parametrize("case", load_approved_cases(...), ids=lambda c: c["id"])
    def test_answer_semantics(case):
        run = run_orchestrator_case(case)
        evidence = extract_final_product_evidence(run)
        test_case = LLMTestCase(
            input=case["input"]["message"],
            actual_output=run.answer,
            retrieval_context=evidence,
        )
        metrics = [
            GEval(
                name="软约束满足度",
                criteria=" ".join(build_soft_constraint_rubric(case)),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                model=ProjectDashScopeJudge(),
                threshold=0.7,
            ),
            GEval(
                name="商品证据忠实性",
                criteria=(
                    "回答中的商品属性必须能从retrieval_context得到支持；"
                    "资料未说明时必须表达不确定，不得补充不存在的参数、成分或功效。"
                ),
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                model=ProjectDashScopeJudge(),
                threshold=0.8,
            ),
        ]
        assert_test(test_case, metrics)
    """


if __name__ == "__main__":
    approved = load_approved_cases(ROOT / "datasets" / "single_turn.jsonl")
    print(
        json.dumps(
            {
                "approved_case_count": len(approved),
                "message": "数据集审核通过后，再把此示例接入 pytest 和真实 Orchestrator。",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
