# hand-location-analysis

会話中の人物を正面から撮影した動画について、各人物の左右の手がどの空間にあったかを自動でアノテーションするツールです。
GUI（Streamlit）とコマンドラインの両方から使えます。

手の位置の定義は Choi, Kim & Chung (2014) "A taxonomy and notation method for three-dimensional hand gestures"
(*International Journal of Industrial Ergonomics*, 44, 171–188) の Fig. 3 の9区画に従います。

本ツールはアノテーション支援ツールです。研究や分析に使う前に、出力動画で結果を必ず確認してください。

## 9区画の定義

撮影の向きに合わせて、正面モード（`--mode frontal`、既定値）と横からモード（`--mode side`）があります。どちらも中央の四角形を通る縦2本・横2本の線で周囲を9区画に分け、手首が入っている区画をその手の位置とします。線の上にある場合は中央側の区画です。左右の肩・腰・膝はそれぞれ左右の中点を使います。

### 正面モード

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
- 四角形は画像の軸に沿った長方形で、体が傾いても回転しません。

### 横からモード

人物の横から撮影し、人物が直角に座っている前提です。

```
            体の前（膝より前）  体と膝の間   体より後ろ
肩より上          1              2            3
肩〜太もも        4              5            6
太ももより下      7              8            9
```

- 縦の線は、体の線（肩の x と腰の x の平均）と膝の線（膝の x）です。
- 横の線は、肩の線（肩の y）と太ももの線（腰の y と膝の y の平均）です。
- 1/4/7 は常に体の前側です。前側は、膝が体の線の左右どちらにあるかで、人物ごとに動画全体で1つに決めます。
- `--orientation` は使いません。

## 処理の流れ

1. Ultralytics YOLO26 の姿勢推定モデル（既定値 `yolo26x-pose`）と BoT-SORT で人物を追跡し、COCO 17 キーポイントを取得します（`poses.csv`）。
2. 人物ごとに、動画の全フレームへ座標と信頼度を補完します。検出の間は線形に補完し、最初の検出より前と最後の検出より後は端の値を保持します。欠損の長さや信頼度によらず補完します（対象の人物はずっと映っている前提です）。
3. `--keypoint-thresh` が0より大きい場合、補完後の信頼度がしきい値を下回るフレームを除外します。手首が下回ったフレームはその手だけ、枠に使うキーポイント（正面モードは肩・腰、横からモードは肩・腰・膝）のどれかが下回ったフレームは両手を除外します。除外したフレームは補完し直しません。
4. 残ったフレームで9区画の枠を作り、左右の手首がどの区画にあるかを判定します。平滑化や短い区間の統合はしません。
5. 区画を分ける線と手首の位置を描いた動画を、元の音声付きで書き出します。

## 現在の構成

- [app.py](app.py): GUI ランチャー
- [handloc/pipeline.py](handloc/pipeline.py): パイプライン全体（`HandLocationAnalyzer`）
- [handloc/tracking.py](handloc/tracking.py): YOLO pose による人物追跡とキーポイント取得
- [handloc/temporal.py](handloc/temporal.py): キーポイントの全フレームへの補完
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
4. `Run` を押すと YOLO で人物を追跡し、検出されたユニークな人物が一覧表示されます（各人物の切り出し画像、track_id、映っていた時間）。
5. 対象にする人物を1人選んで（`Target person`）、`Annotate this person` を押します。選んだ人物だけが以降の処理と出力に進みます。
6. 完了後に、人物・手ごとの区画別の滞在秒数、区間の一覧、出力動画が表示されます。

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
python -m handloc run input.mp4 --output-name session01 --keypoint-thresh 0.5
python -m handloc run input.mp4 --track-id 2   # 2回目以降: 一覧で確認した人物
```

主なオプション:

| オプション | 既定値 | 内容 |
|---|---|---|
| `--output-name` | 入力動画のファイル名 | 出力フォルダの名前。入力動画と同じ場所に作ります（入力が1つのときのみ指定可） |
| `--device` | 利用可能なもののうち最速 | `cuda:0`, `mps`, `cpu` |
| `--pose-model` | `yolo26x-pose` | `yolo26n/s/m/l/x-pose` |
| `--person-thresh` | 0.5 | 人物検出の信頼度しきい値 |
| `--track-id` | 追跡された人物が1人ならその人物 | 対象にする人物（1人）の track_id。YOLO の後に検出された人物の一覧が表示されます。複数の人物が追跡されて指定がない場合は、`poses.csv` と `summary.json` だけを保存して一覧を表示し、終了します。2回目以降に指定すれば YOLO は再実行されません。入力が1つのときのみ指定可 |
| `--keypoint-thresh` | 0 | 補完後、手首・両肩・両腰の信頼度がこれ未満のフレームを除外（0 ですべてのフレームを使用） |
| `--person-fps` | 元動画の FPS | 指定した FPS に間引いて姿勢推定し、間のフレームは補完 |
| `--mode` | `frontal` | `frontal`: 正面モード。`side`: 横からモード |
| `--orientation` | `frontal` | 正面モードのみ。`frontal`: 人物の右側 = 画像の左側。`auto`: 肩の左右の並びからフレームごとに判定 |
| `--no-video` | | 動画を出力しない |
| `--no-cache` | | `poses.csv` を再利用せず、YOLO をやり直す |
| `--track-high-thresh` など | `config/botsort.yaml` | BoT-SORT の設定を上書き |

`python -m handloc run --help` ですべてのオプションを確認できます。

## 出力ファイル

出力フォルダ（入力動画と同じ場所。名前の既定値は入力動画のファイル名）に次のファイルを書き出します。

- `poses.csv`: YOLO の生の結果（選択に関係なく全員分）。補完や推定を含まず、ほかの出力はすべてここから作れます。列は `frame_idx`, `track_id`, `conf`, 人物の枠 `x1`–`y2`, 17キーポイントの `<部位>_x`, `<部位>_y`, `<部位>_conf` です。
- `frames.csv`: 各人物の、各フレームの左右の手首の正規化座標。列は `frame_idx`, `track_id`, `hand`, `x`, `y` です。
  - 正面モード: `x` は右肩の線が 0、左肩の線が 1、`y` は肩の線が 0、腰の線が 1 です。
  - 横からモード: `x` は体の線が 0、膝の線が 1（前向きが正）、`y` は肩の線が 0、太ももの線が 1 です。
  - 外側の区画では 0 未満や 1 超になります。
  - しきい値で除外されたフレームの行は出力しません。
- `locations.csv`: 各人物の左右の手が各区画にあった時間区間。列は `track_id`, `hand`, `startTime`, `endTime`（秒。区間の最後のフレームの終わりの時刻）, `location`（区画番号 1–9）です。除外されたフレームは、どの区間にも含まれません。
- `locations.mp4`: 元の動画に描画したもの（元の音声付き）。すべて半透明です。
  - 人物（track_id）ごとに色を変えて、区画を分ける線と区画番号、9区画全体を囲む外枠を描きます（外枠の範囲は描画用で、判定では外側の区画は外へ無限に続く扱いです）。
  - track_id は外枠の上辺の外側、左上に表示します。
  - 手首は線と同じ色の丸で、右手に R、左手に L を付けます。
  - 手首がある区画だけを、さらに薄く塗りつぶします。塗りは手首ごとに重ねるので、左右の手首が同じ区画なら少し濃くなり、別の人物の区画と重なると色が混ざります。
  - その人物の両手が除外されたフレームでは、track_id だけを描きます。
- `heatmap_right.png`・`heatmap_left.png`・`heatmap_both.png`: 動画全体で手首があった場所のヒートマップ（右手首・左手首・両方の合算）。背景は動画の中央のフレームです。
  - 対象人物の、除外されなかったフレームの手首の位置を数え、ぼかして色付きで重ねます。少ない場所ほど透明です。
  - 色は画像ごとに、その画像の最大値で正規化します（3枚の間で濃さは比べられません）。
  - 対象人物の区画の線と番号を、動画全体での中央値の位置に薄く描きます（目安）。
- `summary.json`: 処理時間、追跡された人物の track_id と対象人物の track_id、人物・手ごとの区画別の滞在秒数と有効フレーム数、設定、`poses.csv` を作った YOLO の設定（`yolo`）。
- `frames.csv`・`locations.csv`・`locations.mp4`・ヒートマップには、対象人物（1人）だけが入ります。

対象人物は1回の実行につき1人です。複数の人物を処理したい場合は、`--output-name` で出力フォルダを分けて1人ずつ実行してください。1回目の出力フォルダの `poses.csv` と `summary.json` を新しいフォルダにコピーしておけば、YOLO は再実行されません。同じ出力フォルダで `--track-id` を変えて実行し直すこともできますが、前の人物の出力は上書きされます。

2回目以降、入力動画・モデル・人物検出のしきい値・FPS・トラッカーの設定が `summary.json` の `yolo` と同じなら、`poses.csv` を再利用して YOLO を省略します。対象人物の選択（GUI の選択・`--track-id`）、`--mode`、キーポイントのしきい値、`--orientation` だけを変えた場合は、YOLO は再実行されません。

## 注意点

- 正面から撮影した動画を前提としています。横向きの人物は左右の肩が重なって中央の区画が細くなるため、結果の解釈に注意してください。
- トラック ID は BoT-SORT の結果です。人物が画面外に出たり、長く隠れたりすると ID が変わることがあります。
- テスト: `python -m pytest tests`
