r"""
测试 LaTeX 公式分割问题

模拟流式输出场景，测试 sentence buffer 如何处理各种 LaTeX 公式：
- \(...\) 内联公式
- $...$ 内联公式
- 公式内部的 ) 和 . 不应导致切分
"""
import sys

# Add ai-service to path
sys.path.insert(0, '/home/zj/ZengKingMorphe/ai-service')

from app.utils.sentence_buffer import SentenceBuffer


# 测试输入：包含多种 LaTeX 公式
TEST_INPUT = r'''
好的，我正在梳理您的问题要点…关于"等比数列"的公司，这个表述似乎有些混淆。通常我们不会直接将某个公司称为"等比数列"，因为这是数学中的概念，并不是公司的名称或特征。

如果您是指与等比数列相关的实际应用案例或者行业，比如金融领域的复利计算，可以考虑这样的场景：

**复利计算示例：**
复利计算是一个典型的等比数列应用。假设您将一笔资金存入银行，按照固定年利率进行复利计息，则每年末的本利和构成一个等比数列。

例如，如果您初始投资1000元，年利率为5%，那么每一年末的资金数额可以表示如下：
- 第1年末：\( 1000 \times (1 + 0.05) = 1000 \times 1.05 \)

- 第2年末：\( 1000 \times (1.05)^2 \)

- 第3年末：\( 1000 \times (1.05)^3 \)

- ...

这个序列中每个相邻项之间的比值都是固定的，即公比 $q = 1.05$。

如果您有具体的问题或需要一个特定行业的例子，可以进一步说明。例如，您是否对金融行业、科技公司或其他领域的应用感兴趣？这样的领域可能会使用等比数列的概念来解决实际问题。
'''

TEST_INPUT2 = '''
等我一下······
当然可以。
以下是几个常用的三角函数公式：


1.
 **基本定义**：
 
   - $\sin \theta = \frac{\text{对边}}{\text{斜边}}$
   
   - $\cos \theta = \frac{\text{邻边}}{\text{斜边}}$
   
   - $\tan \theta = \frac{\sin \theta}{\cos \theta} = \frac{\text{对边}}{\text{邻边}}$
   
   
2.
 **同角三角函数的基本关系**：
 
   - $\sin^2 \theta + \cos^2 \theta = 1$
   
   - $1 + \tan^2 \theta = \sec^2 \theta$
   
   - $1 + \cot^2 \theta = \csc^2 \theta$
   
   
3.
 **和差公式**：
 
   - $\sin (a \pm b) = \sin a \cos b \pm \cos a \sin b$
   
   - $\cos (a \pm b) = \cos a \cos b \mp \sin a \sin b$
   
   - $\tan (a \pm b) = \frac{\tan a \pm \tan b}{1 \mp \tan a \tan b}$
   
   
4.
 **倍角公式**：
 
 
   - $\sin 2\theta = 2 \sin \theta \cos \theta$
   
   
   - $\cos 2\theta = \cos^2 \theta - \sin^2 \theta = 1 - 2 \sin^2 \theta = 2 \cos^2 \theta - 1$
   
   - $\tan 2\theta = \frac{2 \tan \theta}{1 - \tan^2 \theta}$
   
   
5.
 **半角公式**：
 
   - $\sin \frac{\theta}{2} = \pm \sqrt{\frac{1 - \cos \theta}{2}}$
   
   - $\cos \frac{\theta}{2} = \pm \sqrt{\frac{1 + \cos \theta}{2}}$
   
   - $\tan \frac{\theta}{2} = \pm \sqrt{\frac{1 - \cos \theta}{1 + \cos \theta}}$
   
   
6.
 **诱导公式**（利用周期性和奇偶性）：
 
   - $\sin (-\theta) = -\sin \theta$

   - $\cos (-\theta) = \cos \theta$

   - $\tan (-\theta) = -\tan \theta$

   - $\sin (\pi + \theta) = -\sin \theta$

   - $\cos (\pi + \theta) = -\cos \theta$

   - $\tan (\pi + \theta) = \tan \theta$
   
   
这些公式涵盖了三角函数的基本性质和常用变换，
希望对您有所帮助。'
如果您有具体的应用需求或问题，请随时告知！
'''

def test_formula_token_streaming():
    """测试模拟 LLM 逐个 token 输出完整公式的场景"""
    buffer = SentenceBuffer(max_chars=100, max_wait_seconds=0.5)
    segments_collected = []

    print("=" * 80)
    print("测试场景：模拟 LLM 逐个 token 输出完整公式 (TEST_INPUT)")
    print("=" * 80)
    print()

    # 模拟流式输出：逐个字符添加
    for index, char in enumerate(TEST_INPUT):
        segment = buffer.add(char)
        if segment:
            segments_collected.append(segment)

    # 输出最后剩余内容
    final_segment = buffer.buffer
    if final_segment:
        segments_collected.append(final_segment)

    print(f"原始 segments 数量: {len(segments_collected)}")
    print("-" * 80)
    print("原始 segments（处理前）:")
    print("-" * 80)
    for i, seg in enumerate(segments_collected):
        print(f"[{i:2d}] {repr(seg)}")

    # 后处理合并标点 segment
    merged_segments = SentenceBuffer._merge_punctuation_segments(segments_collected)

    print()
    print(f"合并后 segments 数量: {len(merged_segments)}")
    print("-" * 80)
    print("合并后 segments（处理后）:")
    print("-" * 80)
    for i, seg in enumerate(merged_segments):
        print(f"[{i:2d}] {repr(seg)}")

    # 验证：检查是否还有单独的标点 segment
    has_standalone_punctuation = any(
        SentenceBuffer._is_punctuation_only(s) for s in merged_segments
    )
    has_short_prefix = any(
        SentenceBuffer._is_short_prefix_segment(s) for s in merged_segments
    )
    print()
    print(f"验证: 是否存在单独标点 segment = {has_standalone_punctuation}")
    print(f"验证: 是否存在短前缀 segment = {has_short_prefix}")
    if has_standalone_punctuation or has_short_prefix:
        print("❌ 测试失败：仍然存在单独标点或短前缀 segment")
    else:
        print("✅ 测试通过：没有单独标点或短前缀 segment")

    return merged_segments


def test_formula_token_streaming_with_test_input2():
    """测试模拟 LLM 逐个 token 输出完整公式的场景 (TEST_INPUT2)"""
    buffer = SentenceBuffer(max_chars=100, max_wait_seconds=0.5)
    segments_collected = []

    print("=" * 80)
    print("测试场景：模拟 LLM 逐个 token 输出完整公式 (TEST_INPUT2)")
    print("=" * 80)
    print()

    # 模拟流式输出：逐个字符添加
    for index, char in enumerate(TEST_INPUT2):
        segment = buffer.add(char)
        if segment:
            segments_collected.append(segment)

    # 输出最后剩余内容
    final_segment = buffer.buffer
    if final_segment:
        segments_collected.append(final_segment)

    print(f"原始 segments 数量: {len(segments_collected)}")
    print("-" * 80)
    print("原始 segments（处理前）:")
    print("-" * 80)
    for i, seg in enumerate(segments_collected):
        print(f"[{i:2d}] {repr(seg)}")

    # 后处理合并标点 segment
    merged_segments = SentenceBuffer._merge_punctuation_segments(segments_collected)

    print()
    print(f"合并后 segments 数量: {len(merged_segments)}")
    print("-" * 80)
    print("合并后 segments（处理后）:")
    print("-" * 80)
    for i, seg in enumerate(merged_segments):
        print(f"[{i:2d}] {repr(seg)}")

    # 验证：检查是否还有单独的标点 segment
    has_standalone_punctuation = any(
        SentenceBuffer._is_punctuation_only(s) for s in merged_segments
    )
    has_short_prefix = any(
        SentenceBuffer._is_short_prefix_segment(s) for s in merged_segments
    )
    print()
    print(f"验证: 是否存在单独标点 segment = {has_standalone_punctuation}")
    print(f"验证: 是否存在短前缀 segment = {has_short_prefix}")
    if has_standalone_punctuation or has_short_prefix:
        print("❌ 测试失败：仍然存在单独标点或短前缀 segment")
    else:
        print("✅ 测试通过：没有单独标点或短前缀 segment")

    return merged_segments


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("运行 TEST_INPUT 测试")
    print("=" * 80 + "\n")
    test_formula_token_streaming()

    print("\n" * 3)
    print("=" * 80)
    print("运行 TEST_INPUT2 测试")
    print("=" * 80 + "\n")
    test_formula_token_streaming_with_test_input2()
