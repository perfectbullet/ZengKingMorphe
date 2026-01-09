"""
PyPDF vs MinerU 对比分析脚本

对比两种PDF解析方式对RAG切分效果的影响
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_results(file_path: str) -> List[Dict]:
    """加载测试结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(pypdf_results: List[Dict], mineru_results: List[Dict]) -> Dict[str, Any]:
    """对比两种解析方式的结果"""

    # 创建文件名到结果的映射
    pypdf_map = {r['filename']: r for r in pypdf_results if 'error' not in r}
    mineru_map = {r['filename']: r for r in mineru_results if 'error' not in r}

    # 找出共同的文件
    common_files = set(pypdf_map.keys()) & set(mineru_map.keys())

    comparison = {
        'total_files': len(common_files),
        'comparisons': []
    }

    for filename in sorted(common_files):
        pypdf = pypdf_map[filename]
        mineru = mineru_map[filename]

        # 获取最佳配置
        pypdf_best = pypdf.get('recommendation', {})
        mineru_best = mineru.get('recommendation', {})

        # 比较文本长度
        pypdf_len = pypdf.get('text_length', 0)
        mineru_len = mineru.get('text_length', 0)
        len_diff = mineru_len - pypdf_len
        len_pct = (len_diff / pypdf_len * 100) if pypdf_len > 0 else 0

        # 获取各配置的均匀度
        pypdf_configs = {c['config_name']: c for c in pypdf.get('configs', [])}
        mineru_configs = {c['config_name']: c for c in mineru.get('configs', [])}

        comp = {
            'filename': filename,
            'domain': pypdf.get('domain', ''),
            'length_category': pypdf.get('length_category', ''),
            'pypdf_length': pypdf_len,
            'mineru_length': mineru_len,
            'length_diff': len_diff,
            'length_diff_pct': round(len_pct, 1),
            'pypdf_best': pypdf_best.get('best_config', ''),
            'mineru_best': mineru_best.get('best_config', ''),
            'configs_comparison': {}
        }

        # 对比每个配置的均匀度
        for config_name in ['默认(256/50)', '中chunk(512/128)', '超大chunk(800/200)']:
            if config_name in pypdf_configs and config_name in mineru_configs:
                pypdf_uniformity = pypdf_configs[config_name].get('uniformity', 0)
                mineru_uniformity = mineru_configs[config_name].get('uniformity', 0)

                comp['configs_comparison'][config_name] = {
                    'pypdf_uniformity': pypdf_uniformity,
                    'mineru_uniformity': mineru_uniformity,
                    'pypdf_chunks': pypdf_configs[config_name].get('chunk_count', 0),
                    'mineru_chunks': mineru_configs[config_name].get('chunk_count', 0),
                    'winner': 'pypdf' if pypdf_uniformity > mineru_uniformity else 'mineru' if mineru_uniformity > pypdf_uniformity else 'tie'
                }

        comparison['comparisons'].append(comp)

    # 统计汇总
    stats = {
        'text_length': {
            'pypdf_total': sum(c['pypdf_length'] for c in comparison['comparisons']),
            'mineru_total': sum(c['mineru_length'] for c in comparison['comparisons']),
            'avg_diff_pct': sum(c['length_diff_pct'] for c in comparison['comparisons']) / len(comparison['comparisons'])
        },
        'best_config_agreement': {
            'same': sum(1 for c in comparison['comparisons'] if c['pypdf_best'] == c['mineru_best']),
            'different': sum(1 for c in comparison['comparisons'] if c['pypdf_best'] != c['mineru_best'])
        },
        'uniformity_winner': {
            'pypdf': 0,
            'mineru': 0,
            'tie': 0
        },
        'by_length_category': {
            '短文档(<10K)': {'pypdf': 0, 'mineru': 0},
            '中长文档(10-20K)': {'pypdf': 0, 'mineru': 0},
            '超长文档(>20K)': {'pypdf': 0, 'mineru': 0}
        }
    }

    for comp in comparison['comparisons']:
        length_cat = comp['length_category']

        # 统计默认配置的均匀度胜者
        default_comp = comp['configs_comparison'].get('默认(256/50)', {})
        if default_comp:
            winner = default_comp.get('winner', 'tie')
            stats['uniformity_winner'][winner] = stats['uniformity_winner'].get(winner, 0) + 1

            if winner in stats['by_length_category'][length_cat]:
                stats['by_length_category'][length_cat][winner] += 1

    comparison['statistics'] = stats

    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """打印对比报告"""
    print("\n" + "="*80)
    print(" "*25 + "PyPDF vs MinerU 对比报告")
    print("="*80)

    stats = comparison['statistics']

    print(f"\n测试文件数: {comparison['total_files']}")

    print("\n" + "-"*80)
    print("文本长度对比:")
    print("-"*80)
    print(f"PyPDF 总字符数:   {stats['text_length']['pypdf_total']:,}")
    print(f"MinerU 总字符数:  {stats['text_length']['mineru_total']:,}")
    print(f"平均差异:         {stats['text_length']['avg_diff_pct']:+.1f}%")

    print("\n" + "-"*80)
    print("最佳配置一致性:")
    print("-"*80)
    print(f"一致: {stats['best_config_agreement']['same']} 个")
    print(f"不同: {stats['best_config_agreement']['different']} 个")

    print("\n" + "-"*80)
    print("均匀度对比 (默认配置 256/50):")
    print("-"*80)
    print(f"PyPDF 胜出:  {stats['uniformity_winner']['pypdf']} 个")
    print(f"MinerU 胜出: {stats['uniformity_winner']['mineru']} 个")
    print(f"平局:        {stats['uniformity_winner']['tie']} 个")

    print("\n" + "-"*80)
    print("按文档长度分类统计 (默认配置):")
    print("-"*80)
    for length_cat, counts in stats['by_length_category'].items():
        print(f"\n{length_cat}:")
        print(f"  PyPDF:  {counts['pypdf']} 胜")
        print(f"  MinerU: {counts['mineru']} 胜")

    print("\n" + "-"*80)
    print("详细对比 (前10个文件):")
    print("-"*80)

    for i, comp in enumerate(comparison['comparisons'][:10], 1):
        print(f"\n{i}. {comp['filename']}")
        print(f"   长度: PyPDF {comp['pypdf_length']:,} vs MinerU {comp['mineru_length']:,} ({comp['length_diff_pct']:+.1f}%)")
        print(f"   最佳: PyPDF {comp['pypdf_best']} vs MinerU {comp['mineru_best']}")

        default_comp = comp['configs_comparison'].get('默认(256/50)', {})
        if default_comp:
            print(f"   均匀度(256/50): PyPDF {default_comp['pypdf_uniformity']:.4f} vs MinerU {default_comp['mineru_uniformity']:.4f}")
            print(f"   Chunk数(256/50): PyPDF {default_comp['pypdf_chunks']} vs MinerU {default_comp['mineru_chunks']}")

    print("\n" + "="*80)


def main():
    # 加载两种结果
    pypdf_file = "ai-service/docs/rag_chunk_baseline_full.json"
    mineru_file = "ai-service/docs/rag_chunk_baseline_mineru.json"

    print("加载测试结果...")
    pypdf_results = load_results(pypdf_file)
    print(f"PyPDF 结果: {len(pypdf_results)} 个文件")

    try:
        mineru_results = load_results(mineru_file)
        print(f"MinerU 结果: {len(mineru_results)} 个文件")
    except FileNotFoundError:
        print(f"MinerU 结果文件不存在: {mineru_file}")
        print("请先运行: python -m tests.test_rag_batch --output ai-service/docs/rag_chunk_baseline_mineru.json")
        return

    # 进行对比
    print("\n进行对比分析...")
    comparison = compare_results(pypdf_results, mineru_results)

    # 打印报告
    print_comparison_report(comparison)

    # 保存对比结果
    output_file = "ai-service/docs/pypdf_vs_mineru_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
