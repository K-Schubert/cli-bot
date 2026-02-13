# To do
- [x] Refactor project
- [x] Enhance memory management
- [x] Fix tool call strip
- [x] Truncate conversation turns with max_conv_turns
- [ ] PDF upsert: add instruction to upload (trigger llm round) or not (doesnt trigger LLM round if so)
- [ ] Add sample data
- [ ] Setup db collection with HYQ collection + parallel search
- [ ] Metadata search sur doc id, title, url, source
- [ ] Don't systematically call BM25 -> if user uploaded a PDF, only semantic/metadata search and no rerank
- [ ] Add data sources
- [ ] Improve metadata search and semantic search filters
- [ ] Improve prompt to search in uploaded pdf docs/particular docs based on memory/history/user instructions

# How to use
```
cd src
python -m cli.cli_chat --dev-prompt /home/kieran/projects/cli-bot/src/prompts/developer_prompt.md --csv /home/kieran/projects/cli-bot/data/infopers_2.csv --doc-validation filter --verbose --history
```