# Orbbec Gemini 335Le Capture Tool

Orbbec Gemini 335Le 向けのキャプチャ/保存/録画ツール

## 前提
- Python 3.11 以上（uv 使用）
- Orbbec SDK: [pyorbbecsdk v2.0.15](https://github.com/orbbec/pyorbbecsdk/releases/tag/v2.0.15)

## セットアップ

### justfile を使う場合
```bash
# 依存インストール（pyproject.toml に基づき uv sync）
just install

# フォーマット
just format

# 構造テスト（実機不要）
just test-structure

# 実機テスト（接続時のみ）
just test-hw
```

### justfile を使わない場合（手動）
```bash
# 依存インストール（pyproject.toml を使用）
uv sync

# 構造テスト
uv run pytest tests/orbbec/test_structure.py -q

# 実機テスト（カメラ接続時のみ、.venv の Python を明示する方法）
uv run pytest tests/orbbec/test_hardware.py -q --run-hardware
~/.venv/bin/python -m pytest tests/orbbec/test_hardware.py -q --run-hardware

# フォーマット
uv run ruff format .
```

## ツールの使い方（CLI）

`capture_orbbec.py` は OrbbecDevice を用いてフレーム取得し、raw 保存（PNG/NPY+JSON）または .bag 録画を行います。

```bash
# 30フレーム取得して PNG 保存（デフォルトプロファイル）
uv run python capture_orbbec.py --profile default --save-dir ./tmp_out --frames 30

# HDR 有効、露光指定、align=hw で 60 フレーム取得
uv run python capture_orbbec.py --profile hdr --hdr --hdr-exposure1 7500 --hdr-exposure2 50 --align hw --frames 60 --save-dir ./tmp_out

# .bag 録画（save-mode=bag、保存先を指定しない場合は ./output/capture.bag）
uv run python capture_orbbec.py --save-mode bag --bag-path ./tmp_out/capture.bag --frames 120
```

主なオプション:
- `--profile {default,hq,hdr}`: プリセット選択
- `--hdr/--no-hdr`, `--hdr-exposure1`, `--hdr-exposure2`: HDR 設定
- `--align {none,hw,sw}`: アライメント設定
- `--save-dir PATH`: 保存先ディレクトリ
- `--frames N`: 取得フレーム数
- `--save-mode {raw,bag}`: raw は PNG/NPY+JSON 保存、bag は pyorbbecsdk の .bag 録画
- `--bag-path PATH`: .bag の保存先（未指定なら `save-dir/capture.bag`）
- `--no-save`: ファイル保存を無効化（bagモード時は録画のみ）
