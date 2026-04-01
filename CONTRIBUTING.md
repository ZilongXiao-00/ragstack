# 贡献指南

感谢你对 RAGStack 的兴趣！

## 开发环境设置

```bash
git clone https://github.com/yourusername/ragstack.git
cd ragstack
pip install -e ".[dev]"
```

## 代码规范

- 使用 `black` 格式化代码
- 使用 `isort` 排序导入
- 使用 `flake8` 检查代码风格
- 使用 `mypy` 进行类型检查

```bash
black ragstack/
isort ragstack/
flake8 ragstack/
mypy ragstack/
```

## 运行测试

```bash
pytest tests/ -v
```

## 提交 PR

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request
