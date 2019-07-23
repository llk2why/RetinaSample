# python run.py --model_type DemosaicSR
# python run.py --model_type RYYB
# python run.py --model_type Random
# python run.py --model_type Arbitrary
# python run.py --model_type RB_G
# python run.py --model_type RB_G_DENOISE
# python run.py --model_type JointPixel_RGBG

# # Debug
# for i in DemosaicSR RYYB Random RB_G RB_G_DENOISE JointPixel_RGBG ;
# do 
#     for j in 0.02;
#     do
#         python run.py --model_type $i --noise $j --debug
#     done
# done

# Run command
for i in DemosaicSR RYYB Random RB_G RB_G_DENOISE JointPixel_RGBG ;
do 
    for j in 0.02 0.05 0.08 0.1 0.15 0.2 0.25 0.3 0.4 0.5 ;
    do
        python run.py --model_type $i --noise $j
    done
done


