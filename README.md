# LSI-design / LSI-CNN-project

LSIデザインコンテスト向けの小規模CNNです。scikit-learnに同梱されたDigitsの数字 **0・1** を、PyTorchで学習・評価します。Pythonのfloat32モデルと各層の中間値を、今後のVerilog実装との比較基準にします。

## ファイル構成

```text
LSI-CNN-project/
  python/
    model.py          CNN本体と各層の中間出力
    data.py           画素の正規化と学習・評価データの分離
    train.py          学習、重み保存、評価、中間出力保存
    inference.py      保存済み重みの読込と評価データでの推論
  tests/test_cnn.py    演算仕様と保存・再読込のテスト
  requirements.txt    依存ライブラリ
  artifacts/          実行時の生成物（Git管理対象外）
  verilog/            既存のRTL・テストベンチ用ひな形
  weights/weights.mem 既存の空ファイル（固定小数点形式は未定義）
```

## セットアップと実行

Python 3.10以上を使用します。以下はリポジトリのルートから開始するPowerShellの例です。既存の仮想環境があれば作成手順は省略できます。

```powershell
cd LSI-CNN-project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# デフォルト: seed=42、40エポック、評価割合20%、CPUで実行
.\.venv\Scripts\python.exe python/train.py

# チェックポイントに保存された同じ評価データで推論
.\.venv\Scripts\python.exe python/inference.py

# テスト
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOSでは `python3 -m venv .venv` で作成し、実行ファイルを `.venv/bin/python` に読み替えます。データセット本体はscikit-learnに同梱され、学習時のダウンロードはありません。GPUは不要です。

引数は次のように変更できます。出力先が同じ場合、以前の生成物を上書きします。デフォルトの出力先は作業ディレクトリによらず `LSI-CNN-project/artifacts/` です。指定した相対パスは作業ディレクトリから解決します。

```powershell
.\.venv\Scripts\python.exe python/train.py --epochs 60 --batch-size 32 --learning-rate 0.01 --seed 42 --test-size 0.2 --output-dir artifacts/run2
.\.venv\Scripts\python.exe python/inference.py --checkpoint artifacts/run2/model.pt --output-dir artifacts/run2/inference
```

`python -m python.train`、`python -m python.inference` によるモジュール実行も可能です（プロジェクト内から実行）。

## データと学習条件

- 元画像は8×8、1チャネル、画素値は0〜16。固定の定数16で除算し、float32の0〜1にします。評価データから統計量を計算する処理はありません。
- 元データからラベル0・1だけを選択します。計360件（0が178件、1が182件）を、`train_test_split(..., stratify=labels, random_state=seed)` で分割します。デフォルトは学習288件・評価72件で、重複はありません。
- 評価データは学習、モデル選択、早期終了には使用しません。指定エポック数を学習した最終モデルを評価します。
- Python標準乱数、NumPy、PyTorch、DataLoaderの乱数を同じseedで固定します。CPU、1スレッド、`num_workers=0`、決定的アルゴリズムを使用します。同一のライブラリ・実行環境での再現を想定し、環境をまたぐビット単位の一致は保証しません。
- 最適化はAdam（学習率0.01、その他はPyTorchの既定値）、バッチサイズ32、40エポックです。損失は `CrossEntropyLoss`。Linearの生出力（logits）を渡し、Softmax層は追加しません。
- 予測ラベルは `argmax(logits)`。出力0が数字0、出力1が数字1に対応し、同点は数字0を選びます。

## 各層の入出力サイズ

画像の表記は `N×C×H×W`（NCHW）、`N` はバッチ数です。ユーザー指定の `8×8×1` はPyTorchでは `1×8×8` として扱います。

| 層 | 入力 | 出力 | 演算仕様 |
| --- | --- | --- | --- |
| Input | N×8×8（元画像） | N×1×8×8 | float32、画素 / 16 |
| Conv2d | N×1×8×8 | N×4×6×6 | kernel=3×3、stride=1、padding=0、dilation=1、groups=1、biasあり |
| ReLU | N×4×6×6 | N×4×6×6 | 要素ごとの `max(x, 0)` |
| MaxPool2d | N×4×6×6 | N×4×3×3 | kernel=2×2、stride=2、padding=0、ceil_mode=False |
| Flatten | N×4×3×3 | N×36 | チャネル→行→列の順、列が最速で変化 |
| Linear | N×36 | N×2 | 36入力・2出力、biasあり、logitsを出力 |

学習対象はConvの36重み＋4バイアス、Linearの72重み＋2バイアスで、合計 **114パラメータ** です。

## Verilogで再現する演算仕様

以下の添字はすべて0始まりです。数式ではバッチの添字を省略しています。

**Conv2d** はカーネルを反転しない相互相関です。

```text
conv[c,h,w] = conv_bias[c]
            + sum(r=0..2, s=0..2)
              conv_weight[c,0,r,s] * input[0,h+r,w+s]
c=0..3, h=0..5, w=0..5
```

入力範囲外の参照やゼロパディングはありません。ReLUはConvのバイアス加算後に適用します。

```text
relu[c,h,w] = max(conv[c,h,w], 0)

pool[c,h,w] = max(relu[c,2*h+r,2*w+s] for r,s in {0,1})
c=0..3, h=0..2, w=0..2

flat[9*c + 3*h + w] = pool[c,h,w]

logits[k] = fc_bias[k] + sum(i=0..35) fc_weight[k,i] * flat[i]
k=0..1

prediction = 1 if logits[1] > logits[0] else 0
```

重みの配列形状は `conv_weight[4,1,3,3]`、`conv_bias[4]`、`fc_weight[2,36]`、`fc_bias[2]` です。NumPyのC順序で線形化すると最後の添字が最速で変化します。特にLinearの重みは **出力→入力** の順です。

現時点では入力、重み、中間値、出力すべてがIEEE 754 float32の参照モデルです。並列加算・FMAなどによる加算順序の差で微小な誤差が生じる場合があります。固定小数点のビット幅、小数点位置、量子化方法、積と累積器の幅、丸め・飽和・オーバーフローの規則はまだ定義していません。今後それらを決め、固定小数点のPython参照モデルと `.mem` への変換を追加してからRTLと照合してください。

## 保存されるファイル

学習は次の4ファイルを生成します。評価は `model.eval()` と `torch.inference_mode()` で行います。

| ファイル | 内容 |
| --- | --- |
| `artifacts/model.pt` | `model_state_dict`、元Digitsデータ中の学習・評価インデックス、seed、評価割合、正規化定数、クラス対応、形式バージョン |
| `artifacts/weights.npz` | float32の `conv_weight`、`conv_bias`、`fc_weight`、`fc_bias` |
| `artifacts/metrics.json` | 評価精度0〜1、正解数、評価件数、損失、混同行列、学習条件、学習損失履歴、ライブラリバージョン |
| `artifacts/intermediates.npz` | 評価データ全件の各層の出力、正解、予測、元データのインデックス |

推論は `model.pt` の保存済み評価インデックスを再利用します。再分割は行わず、デフォルトでは `artifacts/inference/` に `metrics.json` と `intermediates.npz` を生成します。

`intermediates.npz` のキーと形状は以下です。`T` は評価件数（既定72件）で、各配列の先頭軸は同じサンプル順です。各層への入力は直前の層の出力を参照できます。Convの活性化前出力も独立して保存します。

| キー | 形状 | 型 |
| --- | --- | --- |
| `input` | T×1×8×8 | float32 |
| `conv` | T×4×6×6 | float32 |
| `relu` | T×4×6×6 | float32 |
| `pool` | T×4×3×3 | float32 |
| `flatten` | T×36 | float32 |
| `logits` | T×2 | float32 |
| `targets` | T | int64、正解0/1 |
| `predictions` | T | int64、予測0/1 |
| `dataset_indices` | T | int64、フィルタ前の元Digitsデータにおける位置 |

混同行列は行が正解、列が予測、いずれも `[0, 1]` の順です。NPZはpickle不要で読み取れます。

```python
import numpy as np

with np.load("artifacts/intermediates.npz", allow_pickle=False) as values:
    print(values["conv"].shape)       # (72, 4, 6, 6)
    print(values["logits"][0])       # 最初の評価サンプルの2クラス出力
    print(values["dataset_indices"][0])

with np.load("artifacts/weights.npz", allow_pickle=False) as weights:
    conv_memory_order = weights["conv_weight"].ravel(order="C")
```

## テスト内容

`unittest` で層の形状・パラメータ数、非対称カーネルによるConvの演算方向とバイアス、ReLU・Pool・Flattenの順序、Linearの手計算、argmaxの同点処理、正規化と分割の再現性・非重複、短時間の学習と重みの再読込、保存した中間値・精度・混同行列の一致を確認します。

実行確認（2026-11-10、Windows、Python 3.13.0、PyTorch 2.15.0+cpu、NumPy 2.5.4、scikit-learn 1.9.1）:

- 6件のテストがすべて成功。
- 既定条件の学習と保存済みモデルの推論は、いずれも評価72件中72件正解（100%）。混同行列は `[[36, 0], [0, 36]]`。
- 同じseedで40エポックの学習を再実行し、全重み・全中間出力・評価指標・損失履歴の完全一致を確認。
- 学習時に保存した評価出力と、CLIでチェックポイントを読み直した推論出力の完全一致を確認。

この精度は固定された0/1の評価72件に対する結果であり、他の分割や実データでの精度を保証するものではありません。
