# Kaggle auth: put your token in .kaggle/access_token (gitignored).
.PHONY: data
data:  ## download competition data (wheels, 25 public games, templates)
	mkdir -p data && cd data && KAGGLE_API_TOKEN=$$(cat ../.kaggle/access_token) kaggle competitions download -c arc-prize-2026-arc-agi-3 -p . && unzip -q -o *.zip && rm -f *.zip
