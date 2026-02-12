# To do
- [x] Refactor project
- [x] Enhance memory management
- [x] Fix tool call strip
- [ ] Truncate conversation turns with max_conv_turns
- [ ] Add sample data
- [ ] Setup db collection with HYQ collection + parallel search
- [ ] metadata search sur doc id, title, url, source
- [ ] Don't systematically call BM25 -> if user uploaded a PDF, only semantic/metadata search and no rerank

# How to use
```
cd src
python -m cli.cli_chat --dev-prompt /home/kieran/projects/cli-bot/src/prompts/developer_prompt.md --csv /home/kieran/projects/cli-bot/data/infopers_2.csv --doc-validation filter --verbose --history
```