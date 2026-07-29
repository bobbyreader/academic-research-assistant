"""快速开始示例：演示如何使用科研工作流系统。"""

from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.skill_bridge import SkillStatus
from skills.brainstorming.skill import BrainstormingSkill
from skills.academic_search.skill import AcademicSearchSkill
from skills.statistics.skill import StatisticsSkill
from skills.writing.skill import WritingSkill
from skills.polishing.skill import PolishingSkill
from skills.reviewer.skill import ReviewerSkill


def demo_brainstorming() -> None:
    """演示科学头脑风暴。"""
    print("\n" + "="*60)
    print("【演示 1】科学头脑风暴")
    print("="*60)

    skill = BrainstormingSkill()
    result = skill.execute({
        "research_topic": "AI 对高等教育质量保障的影响",
        "current_stage": "understand_context",
    })

    print(f"状态: {result.status.value}")
    print(f"研究主题: {result.data['research_topic']}")
    print(f"当前阶段: {result.data['current_stage']}")
    print(f"阶段描述: {result.data['stage_description']}")
    print("\n引导问题:")
    for i, prompt in enumerate(result.data["prompts"], 1):
        print(f"  {i}. {prompt}")
    print(f"\n下一阶段: {result.data['next_stage']}")


def demo_academic_search() -> None:
    """演示学术检索。"""
    print("\n" + "="*60)
    print("【演示 2】学术检索")
    print("="*60)

    skill = AcademicSearchSkill()
    result = skill.execute({
        "action": "search",
        "query": "artificial intelligence higher education quality assurance",
        "sources": ["crossref", "pubmed", "semantic_scholar"],
        "max_results": 5,
    })

    print(f"状态: {result.status.value}")
    print(f"检索策略: {result.data['strategy']}")
    print(f"\n检索结果 ({result.data['result_count']} 条):")
    for i, paper in enumerate(result.data["results"], 1):
        print(f"  {i}. {paper['title']}")
        print(f"     作者: {', '.join(paper['authors'])} | 年份: {paper['year']} | 期刊: {paper['journal']}")
        print(f"     DOI: {paper['doi']} | 来源: {paper['source']}")
        print()


def demo_statistics() -> None:
    """演示统计审查。"""
    print("\n" + "="*60)
    print("【演示 3】统计审查")
    print("="*60)

    skill = StatisticsSkill()
    result = skill.execute({
        "action": "audit",
        "statistical_text": "n=5, Student's t-test, p<0.05, data are mean ± SEM",
        "figure_legends": "Figure 1: Bar chart showing treatment effects...",
    })

    print(f"状态: {result.status.value}")
    print(f"整体风险: {result.data['overall_risk']}")
    print(f"\n发现 {result.data['issue_count']} 个问题:")
    for issue in result.data["issues"]:
        print(f"  [{issue['severity'].upper()}] {issue['type']}")
        print(f"    描述: {issue['description']}")
        print(f"    建议: {issue['suggested_fix']}")
        print()

    print("需要作者确认的信息:")
    for item in result.data["author_input_needed"]:
        print(f"  - {item}")


def demo_writing() -> None:
    """演示论文撰写。"""
    print("\n" + "="*60)
    print("【演示 4】论文撰写")
    print("="*60)

    skill = WritingSkill()
    result = skill.execute({
        "task": "abstract",
        "subtask": "structured",
        "payload": {
            "research_topic": "AI 对高等教育质量保障的影响",
            "key_findings": ["效率提升 40%", "风险预测准确率 92%"],
        },
    })

    print(f"状态: {result.status.value}")
    print(f"任务: {result.data.get('task', 'N/A')}")
    print(f"子任务: {result.data.get('subtask', 'N/A')}")
    if "abstract" in result.data:
        abstract = result.data["abstract"]
        print(f"\n摘要结构: {abstract.get('structure', 'N/A')}")
        print(f"字数估计: {abstract.get('word_count', 'N/A')}")
    print(f"\n作者检查项: {result.author_checks}")


def demo_polishing() -> None:
    """演示润色检查。"""
    print("\n" + "="*60)
    print("【演示 5】AI 味检查")
    print("="*60)

    skill = PolishingSkill()
    result = skill.execute({
        "task": "ai_detection",
        "subtask": "ai_tells_check",
        "payload": {
            "text": "This revolutionary study demonstrates that AI leads to unprecedented improvements in education quality. It is widely known that recent advances have proven the game-changing potential of this paradigm shift.",
        },
    })

    print(f"状态: {result.status.value}")
    print(f"任务: {result.data.get('task', 'N/A')}")
    print(f"子任务: {result.data.get('subtask', 'N/A')}")
    if "ai_detection_result" in result.data:
        print("检测到 AI 味问题")
    if result.errors:
        print(f"错误: {result.errors}")
    print(f"作者检查项: {result.author_checks}")


def demo_reviewer() -> None:
    """演示模拟审稿。"""
    print("\n" + "="*60)
    print("【演示 6】模拟审稿")
    print("="*60)

    skill = ReviewerSkill()
    result = skill.execute({
        "task": "nature_criteria",
        "subtask": "originality",
        "payload": {
            "manuscript": "[手稿内容：本研究提出了一种基于 AI 的高等教育质量保障新方法...]",
        },
    })

    print(f"状态: {result.status.value}")
    print(f"任务: {result.data.get('task', 'N/A')}")
    print(f"评估维度: {result.data.get('dimension', 'N/A')}")
    if "assessment" in result.data:
        assessment = result.data["assessment"]
        print(f"评级: {assessment.get('rating', 'N/A')}")
    if result.errors:
        print(f"错误: {result.errors}")
    print(f"作者检查项: {result.author_checks}")


def main() -> None:
    """运行所有演示。"""
    print("\n" + "="*60)
    print("  Nature Skills 科研工作流系统 - 快速开始演示")
    print("="*60)

    demo_brainstorming()
    demo_academic_search()
    demo_statistics()
    demo_writing()
    demo_polishing()
    demo_reviewer()

    print("\n" + "="*60)
    print("  演示完成！")
    print("="*60)
    print("\n下一步:")
    print("  1. 编辑 config/settings.yaml 配置 API 密钥")
    print("  2. 运行 python orchestrator.py init <项目名> --mode hybrid")
    print("  3. 运行 python orchestrator.py run <项目名>")
    print("  4. 查看 README.md 获取完整文档")
    print()


if __name__ == "__main__":
    main()
