# EXPERIMENT_NOTES.md

実験固有の注意事項をIDごとに記録する。plan-next-experimentでplan.mdを書く際、
review-expで結果を集約する際、debug-experimentで調査する際は、対象実験のIDに
該当する節があればまず読むこと。

## 0002_20260918_eval_pathology_fms_organ_probe

0001で使っていたVIPERデータセット(`data/datasets--MahmoodLab--viper/`、419
ユニーク画像・`organ`列9クラス)を、VLMではない病理特化feature extractor
5種(UNI, CONCH, Virchow2, H-optimus-0, および比較対象のImageNet教師あり
学習ResNet50)の組織分類プローブ評価に転用する実験。0001のVLM評価コード
(vLLM serve + `viper-eval`)には手を入れず、完全に別の`experiment.py`
(素のPyTorch推論、サーバなし)を新規に書いた。

### 結果サマリ(2026-09-19、5モデル完走・`compare_results.py`で再生成可能)

StratifiedKFold 5分割、linear probe accuracy:
conch 0.945 > virchow2 0.919 > uni 0.909 ≈ h-optimus-0 0.907
>> resnet50-imagenet(ImageNetベースライン) 0.788

**病理特化の事前学習は明確に効いている**(4モデルとも0.90超、ImageNet
ベースラインとの差は12〜16pt)。4モデル間の差は相対的に小さい。

`analyze_errors.py`でモデル横断のOOF予測誤りを集計したところ(419枚中
311枚=74%は5モデル全て正解、9枚は5モデル全て不正解):
- **臓器によって誤り率が大きく偏る**: heart(41%)、gastrointestinal_tract/
  lung(20%)、salivary_gland(19%)、male_reproductive_system(18%)が
  誤りやすい一方、liver(1.6%)、thyroid(2.5%)、urinary_bladder(5.8%)、
  kidney(8.8%)はほぼ全モデルが正解する。**病理特化かImageNetかを問わず
  同じ臓器で同じように間違えている**(=モデル固有の癖ではなく、画像/
  タスク側の難しさ)。
- 5モデル中4つ以上が外した21枚は、`category`が`identify_anatomy`/
  `localize_in_image`(=解剖部位の特定・局在同定)に偏り、
  `magnification`は20x(高倍率、視野が狭く臓器全体の文脈が見えない)が
  大半を占める。低倍率(2.5x)ほど誤り率が低い傾向とも整合的。
  **解釈**: 肝臓/腎臓/甲状腺は高倍率の一視野でも認識できる特徴的な
  組織構造(肝索・糸球体・濾胞)を持つが、心臓/消化管/唾液腺/雄性生殖器系
  は視野を切り取ると他組織と紛らわしくなりやすい、という仮説と整合する。
- `source`列は`TG-GATEs`が誤り0%・`MMO`に誤りが集中しているが、
  `TG-GATEs`は(Open TG-GATEsが元々ラット肝臓の毒性データベースのため)
  ほぼ`liver`のみに対応しており、**臓器の効果と交絡している**点に注意
  (source起因の効果とは言い切れない)。
- `category`も同様に、画像の重複排除で「その画像に紐づく最初の1問」の
  値を使っているため、必ずしもその画像自体の難易度を代表するとは限らない
  (交絡の可能性がある点は留意)。
- 詳細は`outputs/0002_.../error_analysis.csv`(画像ごとのモデル別正誤+
  メタデータ)。

### 設計メモ

- `organ`列のクラス別画像数は最少で19枚(salivary_gland)。`config.yml`の
  `n_splits: 5`のStratifiedKFoldはこの前提で選んでいる(2026-09-18、
  実データで確認済み)。
- 4モデルの重みは`data/models--<org>--<name>/...`にVIPERデータセットと
  同じHF hubキャッシュ形式で置いてある(2026-09-19、`/workspace/andre01/`
  の別プロジェクトの既存キャッシュ(gatedアクセス承認済み)からコピー)。
  **`hf_cache_dir`という独立の設定ではなく、`dataset_dir`(=`data/`)を
  そのまま`HF_HUB_CACHE`として使う**設計にした(`HF_HUB_OFFLINE=1`と
  併用)。当初は`/workspace/andre01/...`を指す`hf_cache_dir`をconfig.ymlに
  持たせて外部キャッシュを直接参照する設計だったが、**grace01は
  `/workspace/andre01`をマウントしておらずジョブから一切参照できない
  ことが実機投入で判明した**(`/workspace/grace01`が他ノードから見えない
  のと対称的に、grace01からも他ノードのworkspaceは見えない。ノード間で
  共有されているのは`/workspace/filesrv02`のような本物のNFSサーバ上の
  領域だけで、`/workspace/<ノード名>`はそのノード自身のローカル領域)。
  そのため重みを`data/`配下にコピーする方式に変更した。
- Virchow2/H-optimus-0は`timm.create_model("hf-hub:...")`でロードする。
  UNI/CONCH/resnet50-imagenetは`models.yml`の`weights_glob`でキャッシュ内の
  重みファイルを直接globして`state_dict`を読む(hf-hub経由にしていない)。
  resnet50-imagenet(timmの"resnet50.tv_in1k"タグ)はローカルキャッシュに
  config.jsonが無く"hf-hub:"文字列でのロードが効かなかったため、UNIと同じ
  直接ロード方式にした(`timm.create_model("resnet50", pretrained=False)`
  を素で構築し、`safetensors.torch.load_file`でstate_dictを流し込む、
  `strict=False`で分類ヘッド分のキー不一致を許容)。
- CONCHの専用pipパッケージ(`conch @ git+https://github.com/Mahmoodlab/CONCH.git`)
  は`uv sync`で問題なく解決できた(2026-09-19、`conch==0.1.0`としてインストール、
  依存の`ftfy`/`h5py`も自動解決)。
- **`SIF_PATH`のハマりどころ**: `scripts/slurm_entry.sh`の`.env`パーサは
  「環境に既に値がある変数は上書きしない」設計だが、このクラスタの
  シェル環境には(由来不明・おそらく別プロジェクト作業時の残留)
  `SIF_PATH`が既に別プロジェクト(`01-toxpatho/toxpatho-ssl-comparison`)
  の存在しないファイルを指した状態で乗っていることがある。この状態だと
  `.env`に正しい`SIF_PATH`を書いてもジョブには反映されず、
  `run_slurm.sh`はapptainerなしのホスト実行にサイレントに
  フォールバックしてしまう(実害は無いが、apptainer経由での実行を
  意図している場合は要注意)。確実に効かせるには、sbatch投入する
  シェルで明示的に`export SIF_PATH=<正しいパス>`してから投入すること。
- viper-benchmark自体には`.sif`/`.def`が無かったため、`00-utils/
  data-io-test/env/env.def`(グループ共通の「uvのみ入った最低限の環境」
  テンプレート、CUDA 12.8.1-cudnn-devel-ubuntu24.04ベース)をそのまま
  `env/env.def`としてコピーして`make build_sif p=<partition>`でビルドした
  (2026-09-19、`env/env.sif`、5.7GB)。**grace01はaarch64(NVIDIA Grace)
  ノードで、ビルドログにも`libboost-fiber-dev:arm64`等arm64パッケージが
  並ぶ**。`pyproject.toml`の`torch==2.11.0+cu130`はこの構成で問題なく
  解決できている(0001が既に動いていた実績通り)。
- `experiment.py`の4モデル全て(uni/conch/virchow2/h-optimus-0)の
  ロード・埋め込み抽出・組織分類CVを実機(grace01、job 11050/11051)で
  検証済み(2026-09-19)。埋め込み次元は設計通り: uni=1024, conch=512
  (`models.yml`の仮置き値と一致), virchow2=2560(CLS+mean-patch結合),
  h-optimus-0=1536。結果(StratifiedKFold 5分割、linear probe accuracy):
  conch 0.945 > virchow2 0.919 > uni 0.909 > h-optimus-0 0.907
  (詳細は`outputs/0002_.../comparison.csv`、`compare_results.py`で再生成可能)。

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
