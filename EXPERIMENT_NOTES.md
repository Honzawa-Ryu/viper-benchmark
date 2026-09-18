# EXPERIMENT_NOTES.md

実験固有の注意事項をIDごとに記録する。plan-next-experimentでplan.mdを書く際、
review-expで結果を集約する際、debug-experimentで調査する際は、対象実験のIDに
該当する節があればまず読むこと。

## 0001_20260907_eval_public_models_viper

VIPER (arxiv.org/abs/2608.26382, 獣医病理VLMベンチマーク) の再現。ただし自前学習は
行わず、論文の16モデルのうち「重みまたはAPIが公開されている」ものだけを
`viper-eval`（mahmoodlab/viperのハーネス、`pyproject.toml`にgit依存として追加済み）
で評価する。

### 未解決のブロッカー（着手前に必要な作業。すべてユーザー側の対応が必要）

1. **データセットがHFでgated（自動承認）**: `MahmoodLab/viper` は
   `datasets.load_dataset` / `viper-eval` 実行時に自動でダウンロードされるが、
   事前に https://huggingface.co/datasets/MahmoodLab/viper を自分のHFアカウントで
   開いて利用規約に同意する必要がある（2026-09-07時点でトークン未承認を確認済み）。
   承認後、そのアカウントのアクセストークンを `.env` の `HF_TOKEN` に設定すること。
2. **LLM judgeにAPIキーが要る**: 自由回答形式(free-text)の採点は
   `viper-eval` 内部でLLM judge（デフォルト `gpt-5.4`）を呼ぶ。ローカルモデルのみを
   評価する場合でも、judgeだけは何らかのAPI（またはローカルにサーブした代替judge、
   `config.yml` の `judge_api_base`）が必要。`.env` の `OPENAI_API_KEY` 等を設定すること。

### 評価対象外にしたモデル

- `ToxScribe`（Qwen3.5版/Gemma4版）: 論文独自の新規モデル。2026-09-07時点で重み公開の
  情報を確認できず。
- `PathChat+`: 著者への直接依頼が必要（HF/GitHubでの一般配布なし）。

### 設計メモ

- `models.yml`（実験ディレクトリ直下）が評価対象モデルのレジストリ。
  `served_locally: true` のモデルは `experiment.py` がジョブ内で `vllm serve` を
  自動起動し、`served_locally: false` のモデルは対応するAPIキー(`api_key_env`)を
  読んでAPI経由で叩く。
- GPUが1基しかないため `RUN_MODE="seq"` で複数モデルを1ジョブ内で順番に評価する
  （並列実行はしない）。seqループは1モデルの失敗で即停止するので、
  依存が揃っているモデルから `run_slurm.sh` の `GRID_VALUES` に並べること。
- `llava-med-7b`（`model_type=llava_mistral` がtransformers/vLLMで未認識との報告あり）
  と `pathgen-llava`（カスタムvision encoder）は動作未検証。まず
  `patho-r1-3b` / `patho-r1-7b` / `quilt-llava-7b` / `medgemma-4b-it` で
  パイプライン疎通確認すること。
- `config.yml` の `limit: 5` はスモークテスト用。本番評価(全1,251問)に進む際は
  `null` にする。
- `tools/create_exp.sh` の owner推定（`/workspace/<owner>/*` → partition）に
  `/workspace/filesrv02/*` → `grace-i` のケースを追加済み（元々このマシンの
  実パスが未対応だった上、grace01の実パーティション名は`grace01`ではなく
  `grace-i`サフィックス付きだったため、`/workspace/grace01/*`側の既存マッピングも
  合わせて修正）。`scripts/preflight_check.py` の `check_config_experiment_match`
  内の正規表現も既存の構文エラー（括弧の対応漏れ）を修正済み（本実験固有ではなく、
  テンプレート共通の既存バグだった）。
- **grace01は`/scratch`が書き込み不可（Permission denied）**。`run_slurm.sh`の
  `USE_LOCAL_SSD_INPUT`/`USE_LOCAL_SSD_OUTPUT`はテンプレートのデフォルトが`1`だが、
  このマシン上ではジョブが`mkdir /scratch`で即失敗する。`toxpatho-vlm`（同じマシン上の
  別実験リポジトリ）でも同じ理由で全実験`0`にしていたのを確認。grace01向けの新規実験は
  作成時に必ず`0`に変更すること（テンプレート側のデフォルトは他マシン向けの可能性が
  あるため、共有ファイルの方は変更していない）。
- **grace01にはpython3-dev(Python.h)が入っておらず、`vllm_enforce_eager: true`
  にしていてもvLLM serveが毎回起動時に落ちる**（2026-09-08、`llava-med-7b`で確認）。
  `enforce_eager`はtorch.compile/CUDAGraphsをスキップするだけで、サンプラー
  (top-k/top-p)のTritonカーネルはeagerモードでも常にJITコンパイルされるため、
  gccが`Python.h`を見つけられず
  `fatal error: Python.h: そのようなファイルやディレクトリはありません` →
  `RuntimeError: vllm serve exited early with code 1`になる。
  sudoが無い（パスワード必須でNG）ため、`apt-get download python3.12-dev
  libpython3.12-dev`でdebだけ取得し`dpkg-deb -x`で展開した（ミラーにはピン留め
  したい厳密パッチバージョンが無いことがあるので、`apt-cache policy`で出てくる
  もう少し古いベースのバージョン文字列、例えば`3.12.3-1`を指定すると取得できた）。
  展開したヘッダは`/workspace/grace01/honzawa/venvs/python3.12-dev-headers/`
  （venvと同じ階層、gitリポジトリ外）に置き、`.env`の`CPATH`でgccに見せている。
  **`.env`は`slurm_entry.sh`内で行単位の簡易パーサ(`KEY=VALUE`)で読まれ、
  bashの`source`ではないので`${VAR:+...}`等のシェル展開は効かない**（値は
  リテラル文字列としてそのままexportされる）。`.env`に変数展開を書きたくなったら
  要注意。今後このマシンで新規にローカルvLLMモデルを追加する実験でも同じ現象が
  起きるはずなので、このワークアラウンドは`viper-benchmark`固有ではなくgrace01
  共通の問題として扱ってよい。
- **`viper-eval`の`--output`は単一ファイルではなく「結果ツリーのルート
  ディレクトリ」**（実データは`<output>/<model_dir_name>/<timestamp>/
  results.json`に書かれる）。`experiment.py`側で`run_dir / "viper_results.json"`
  という"ファイル名"をそのまま渡していたため、viper-evalがそのパスを
  ディレクトリとして`mkdir`し、後続の`Path.read_text()`が
  `IsADirectoryError`で落ちていた（2026-09-12、job 10635で確認）。
  `experiment.py`側を`run_dir / "viper_eval"`という素直なディレクトリに直し、
  `viper_output_dir.glob("*/*/results.json")`で実ファイルを探すように修正済み。
- **`gemini-3.6-flash`は無料枠のレート制限（5 req/min）にすぐ抵触する**
  （2026-09-12、job 10640で確認: `429 RESOURCE_EXHAUSTED`）。`viper-eval`の
  リトライは最大4回・指数バックオフ上限20秒までしかなく、5RPMの枠を
  超えたリクエストは吸収しきれず失敗する。加えて`config.yml`は
  judge_model/extract_modelもgemini-3.6-flashを共用しているため、
  gemini-3.6-flashをモデル一覧から外していても、本番規模(`limit: null`)で
  judge呼び出しの頻度が上がれば同じ枠に当たる可能性がある。
  「Google One AI Premium(Gemini Advanced)」等の個人向けサブスクとは無関係
  （それはGemini appやWorkspace向けの契約で、API利用のレート制限には効かない）。
  解消するにはGoogle AI Studioでこのキーの紐づくプロジェクトにCloud Billing
  を有効化する必要がある（2026-09-13時点でユーザー判断により保留、
  `run_slurm.sh`のGRID_VALUESからは一旦除外）。
- **上記の懸念は的中した**: `llava-med-7b`の本番実行(`limit: null`、job 10680、
  2026-09-14)はクラッシュせず`completed`になったが、`viper_eval_stderr.log`を
  見ると judge/extract 呼び出しの大半がレート制限で失敗している
  （`Judge call failed`: 418件中393件=94%、`MCQ extraction failed`/
  `MCQ LLM extraction failed`: 2095ローテーション中72件、
  `TF LLM extraction failed`: 41件）。`viper-eval`はjudge/extract失敗時に
  例外を投げず`judge_score=None`等にして処理を継続する設計
  (`viper/inference.py`の`_judge`)なので、ジョブは正常終了扱いになり
  `results.json`にも`overall_score`等の数値は出力される。しかし
  `free_text_judge`キーは一件もスコアが集まらず出力されず、
  `overall_score`の計算は代わりに弱い代替指標`free_text_rouge`に
  フォールバックしている(`viper/scoring.py`の`score()`/`_overall_from_metrics`)。
  **つまり現状のoutputs/llava-med-7b/results.jsonの数値は論文と同じ手法の
  スコアではなく、信頼できる評価値として扱ってはいけない**。本番評価を
  やり直す前に、Gemini側のCloud Billing有効化（またはjudge/extractに
  レート制限の緩い別プロバイダを使う）を先に済ませること。
