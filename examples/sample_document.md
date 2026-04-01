# 示例文档

这是 RAGStack 的示例文档。

你可以将任何 Markdown 或纯文本文件作为知识库，RAGStack 会自动进行分片、索引和检索。

## 使用步骤

1. 准备你的文档（支持 .md, .txt 等格式）
2. 调用 `rag.build_index("your_file.md")` 构建索引
3. 调用 `rag.query("你的问题")` 获取回答

## 示例问题

- "文档的主要内容是什么？"
- "RAGStack 支持哪些功能？"
- "如何使用自定义配置？"
