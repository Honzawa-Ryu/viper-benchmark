# 実験 Example カタログ

新規に実験を作成する際、類似の実験があればこれらの example のコード構成（`experiment.py` / `config.yml` / `run_slurm.sh` の分担設計など）をベースライン（お手本）として参照し、新しい実験コードを組み立てます。

---

## 📋 登録されている Example 一覧

現時点で登録されているベストプラクティス・実例の一覧です。

| Example名 | 用途・説明 | 含まれる要素 | 参照パス |
| :--- | :--- | :--- | :--- |
| `baseline_gemma_eval` | LLM（Gemmaなど）を用いた Zero-shot 評価実験の基本構成 | - HuggingFace/vLLM推論<br>- テストデータセット評価ループ<br>- config.yml でのモデル/タスク指定 | `templates/examples/baseline_gemma_eval/` |
| `data_preprocess` | 実験データの加工・フィルタリング・前処理の基本構成 | - Parquetファイルの入出力<br>- データ拡張・クレンジング | `templates/examples/data_preprocess/` |

---

## 📝 Example の追加・利用規約
- **クリーンであること**: Example フォルダ内には、実行ログや outputs などのキャッシュファイルを含めないでください。`experiment.py`, `config.yml`, `run_slurm.sh` (または `plan.md`) のみで構成します。
- **動作確認済みであること**: 必ず `COMPLETED` ステータスになり、テストをパスした実績のあるバグのない実験コードのみを登録します。
