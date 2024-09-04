# First Order Motion Model for Mona Lisa
このプロジェクトはwebカメラを使用して、自分の顔の動きに連動させ、モナリザの顔を動かすことが目的である。

## 使用方法
1.レポジトリのクローン
2.pip install -r requirements.txt
3.FOMMの公式のgithubページから、vox-cpk.pth.tarをルートディレクトリにダウンロード
4.docker build -t first-order-model . で docker image　を作成
5.docker run --gpus all -it --rm -p 8000:8000 first-order-model でコンテナを動かす
6.capture.pyを実行する

## 変更点
本プロジェクトは[First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model)を基に作成している。
主な変更点はcapture.py、main.pyの作成とdemo.pyにAnimationMakerクラスを追加したことである。

## 注意点・問題点
・capture.pyはwindowsで、コンテナの作動はwslで行うことを想定
・GPUを使用する必要がある
※使用してみたところ、様々な点が不十分であり、目や口の動きに反応してモナリザの顔が動きはするが、生成されたモナリザの顔は不安定なものである。

## Credits
This project is based on the [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) by Aliaksandr Siarohin.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The original FOMM project is also licensed under the MIT License. The original copyright notice is retained:
