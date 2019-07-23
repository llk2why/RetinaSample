python run.py --model_type DemosaicSR
python run.py --model_type RYYB
python run.py --model_type Random
python run.py --model_type Arbitrary
python run.py --model_type RB_G
python run.py --model_type RB_G_DENOISE
python run.py --model_type JointPixel_RGBG

python run.py --model_type DemosaicSR --noise 0.05
python run.py --model_type RYYB --noise 0.05
python run.py --model_type Random --noise 0.05
python run.py --model_type RB_G --noise 0.05
python run.py --model_type RB_G_DENOISE --noise 0.05
python run.py --model_type JointPixel_RGBG --noise 0.05

python run.py --model_type DemosaicSR --noise 0.1
python run.py --model_type RYYB --noise 0.1
python run.py --model_type Random --noise 0.1
python run.py --model_type RB_G --noise 0.1
python run.py --model_type RB_G_DENOISE --noise 0.1
python run.py --model_type JointPixel_RGBG --noise 0.1

python run.py --model_type DemosaicSR --noise 0.2
python run.py --model_type RYYB --noise 0.2
python run.py --model_type Random --noise 0.2
python run.py --model_type RB_G --noise 0.2
python run.py --model_type RB_G_DENOISE --noise 0.2
python run.py --model_type JointPixel_RGBG --noise 0.2

python run.py --model_type DemosaicSR --noise 0.5
python run.py --model_type RYYB --noise 0.5
python run.py --model_type Random --noise 0.5
python run.py --model_type RB_G --noise 0.5
python run.py --model_type RB_G_DENOISE --noise 0.5
python run.py --model_type JointPixel_RGBG --noise 0.5

python run.py --model_type DemosaicSR --noise 0.5 --noise 0.5 --debug True
python run.py --model_type RYYB --noise 0.5 --debug True
python run.py --model_type Random --noise 0.5 --debug True
python run.py --model_type Arbitrary --noise 0.5 --debug True
python run.py --model_type RB_G --noise 0.5 --debug True
python run.py --model_type RB_G_DENOISE --noise 0.5 --debug True
python run.py --model_type JointPixel_RGBG --noise 0.5 --debug True