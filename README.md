# hand-location-analysis

会話中の人物を正面から撮影した動画について、各人物の左右の手がどの空間にあったかを自動でアノテーションするツールです。
GUI（Streamlit）とコマンドラインの両方から使えます。

手の位置の定義は Choi, Kim & Chung (2014) "A taxonomy and notation method for three-dimensional hand gestures"
(*International Journal of Industrial Ergonomics*, 44, 171–188) の Fig. 3 の9区画に従います。

本ツールはアノテーション支援ツールです。研究や分析に使う前に、出力動画で結果を必ず確認してください。

## 9区画の定義

```
            人物の右側   中央        人物の左側
肩より上      1          2           3
肩〜腰        4          5（胴体）    6
腰より下      7          8           9
```

- 中央の区画5は、左右の**肩**キーポイントを横の範囲、肩のラインから**腰**のラインまでを縦の範囲とする四角形です。
- この四角形の辺を延長した縦2本・横2本の線で、周囲を9区画に分けます。
- 区画番号は論文と同じく**本人から見た向き**です。正面から撮影した動画では、人物の右側（1/4/7）は画像の左側に映ります。
- 各手の位置は、その手の**手首**キーポイントが入っている区画です。線の上にある場合は中央側の区画とします。
- 左右の手（`right` / `left`）は本人にとっての右手・左手です。
- 腰が机などで隠れて検出できないフレームでは、同じ人物で腰が見えていたフレームの「(腰の高さ − 肩の高さ) / 肩幅」の中央値から腰のラインを推定します。腰が一度も見えない人物には `--hip-ratio`（既定値 1.3）を使います。推定した腰のラインは、出力動画では破線で描かれます。

## 処理の流れ

1. Ultralytics YOLO26 の姿勢推定モデル（既定値 `yolo26x-pose`）と BoT-SORT で人物を追跡し、COCO 17 キーポイントを取得します。
2. 信頼度の低いキーポイントは欠損として扱い、短い欠損は線形補間したうえで移動平均で平滑化します。
3. フレームごとに9区画のグリッドを作り、左右の手首がどの区画にあるかを判定します。
4. 区画の境界付近でのちらつきを抑えるため、短い区間は前後の区間に統合してから、時間区間の CSV に書き出します。
5. 区画を分ける線と手首の位置を描いた動画を、元の音声付きで書き出します。

## 現在の構成

- [app.py](app.py): GUI ランチャー
- [handloc/pipeline.py](handloc/pipeline.py): パイプライン全体（`HandLocationAnalyzer`）
- [handloc/tracking.py](handloc/tracking.py): YOLO pose による人物追跡とキーポイント取得
- [handloc/temporal.py](handloc/temporal.py): キーポイントの補間と平滑化
- [handloc/locations.py](handloc/locations.py): 9区画の判定と時間区間の作成
- [handloc/annotate.py](handloc/annotate.py): フレームへの描画
- [handloc/exporters.py](handloc/exporters.py): CSV と動画の出力
- [handloc/config.py](handloc/config.py): 設定、デバイス、動画メタデータ
- [handloc/cli.py](handloc/cli.py): CLI
- [handloc/gui.py](handloc/gui.py): Streamlit GUI
- [config/botsort.yaml](config/botsort.yaml): BoT-SORT トラッカー設定

## 動作環境

- Python 3.10
- `ffmpeg`（システムの `ffmpeg`、または `ffmpeg/mac/ffmpeg` などの同梱版）
- NVIDIA GPU（`cuda:N`）、Apple Silicon（`mps`）、または CPU

## インストール

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CUDA を使う場合は、先に CUDA 版の `torch` / `torchvision` を PyTorch のインデックスから入れてください。

姿勢推定モデルは、初回使用時に `~/.handloc/` へ自動でダウンロードされます。

## GUI の使い方

```bash
python app.py
```

または

```bash
python -m handloc gui
```

`start.sh` / `start.bat` は、同梱の `venv` を使って同じことをします。
ポートは既定で 8501 を使い、使用中なら 8502、8503… と空いているポートを自動で使います。ポートを指定する場合は `--port 8600` を付けます。

1. `Input video` に動画のパスを入力するか、`Browse` で選びます（macOS / Windows）。
2. 必要に応じて `Output folder name` と `Device` を変更します。
3. 必要に応じて `Detailed Settings` を開きます。
4. `Run` を押します。
5. 完了後に、人物・手ごとの区画別の滞在秒数、区間の一覧、出力動画が表示されます。

### サーバーで使う場合

```bash
# サーバー側
./start.sh --port 8600
# 手元のマシン
ssh -L 8600:localhost:8600 user@server
```

手元のブラウザで `http://localhost:8600` を開きます。入力パスはサーバー上のパスを指定します。

## CLI の使い方

```bash
python -m handloc run input.mp4
python -m handloc run a.mp4 b.mp4 --device cuda:0
python -m handloc run input.mp4 --output-dir out/ --min-segment 0.3 --no-video
```

主なオプション:

| オプション | 既定値 | 内容 |
|---|---|---|
| `--output-dir` | `<入力と同じフォルダ>/<入力のファイル名>` | 出力先（入力が1つのときのみ指定可） |
| `--device` | 利用可能なもののうち最速 | `cuda:0`, `mps`, `cpu` |
| `--pose-model` | `yolo26x-pose` | `yolo26n/s/m/l/x-pose` |
| `--person-thresh` | 0.5 | 人物検出の信頼度しきい値 |
| `--keypoint-thresh` | 0.5 | これ未満の信頼度のキーポイントは欠損扱い |
| `--person-fps` | 元動画の FPS | 指定した FPS に間引いて姿勢推定し、間のフレームは補間 |
| `--smoothing-window` | 5 | キーポイントの移動平均の窓（フレーム数） |
| `--max-gap` | 0.5 | この秒数以下のキーポイント欠損を補間 |
| `--min-track` | 1.0 | 検出時間がこの秒数未満のトラックは除外 |
| `--min-segment` | 0.2 | この秒数未満の区間は前後の区間に統合（0 で無効） |
| `--hip-ratio` | 1.3 | 腰が一度も見えない人物に使う (腰 − 肩) / 肩幅 |
| `--orientation` | `frontal` | `frontal`: 人物の右側 = 画像の左側。`auto`: 肩の左右の並びからフレームごとに判定 |
| `--no-video` | | 動画を出力しない |
| `--no-cache` | | `poses.csv` を再利用せず、姿勢推定をやり直す |
| `--track-high-thresh` など | `config/botsort.yaml` | BoT-SORT の設定を上書き |

`python -m handloc run --help` ですべてのオプションを確認できます。

## 出力ファイル

出力先フォルダ（既定値は入力動画と同じ場所の `<入力のファイル名>/`）に次のファイルを書き出します。

- `hand_locations.csv`: 各人物の左右の手が各区画にあった時間区間
  - `track_id`: 人物のトラック ID
  - `hand`: `right` / `left`
  - `startTime`, `endTime`: 秒（`endTime` は区間の最後のフレームの終わりの時刻）
  - `location`: 区画番号 1–9
  - 手首が検出できない時間は、どの区間にも含まれません。
- `hand_locations.mp4`: 区画を分ける線、区画番号、手首の位置（赤 = 右手、青 = 左手）と現在の区画を描いた動画（元の音声付き）
- `frames.csv`: フレームごとの平滑化済みキーポイント、グリッドの座標、判定結果（`<hand>_location_raw` は統合前、`<hand>_location` は統合後）
- `poses.csv`: YOLO の生の検出結果（キャッシュ）
- `summary.json`: 設定、処理時間、人物・手ごとの区画別の滞在秒数

入力動画、モデル、人物検出のしきい値、FPS、トラッカー設定が同じ場合、2回目以降は `poses.csv` を再利用して姿勢推定を省略します。区画判定の設定だけを変えて何度でもやり直せます。

## 注意点

- 正面から撮影した動画を前提としています。横向きの人物は左右の肩が重なって中央の区画が細くなるため、結果の解釈に注意してください。
- トラック ID は BoT-SORT の結果です。人物が画面外に出たり、長く隠れたりすると ID が変わることがあります。
- テスト: `python -m pytest tests`
