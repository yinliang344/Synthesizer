# Synthesizer
a new function for sequence
根据自己的理解，代码实现了synthesizer
有些地方可能理解不到位
# 疑惑的地方
1、为什么隐藏层的维度要和序列长度一致？在transformer中并不要求隐藏的长度
2、低秩分解random synthesizer模型中，要求两个矩阵的最后一个维度相同吗？具体的处理方法是什么？
3、文章说支持多头注意力机制，但是有些地方并没有想明白如何支持，我实现的代码中的方法是传统multi-head attention的方法，不知道是否正确。
