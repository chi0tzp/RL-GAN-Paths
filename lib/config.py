########################################################################################################################
## Basic configuration file.                                                                                          ##
##                                                                                                                    ##
##                                                                                                                    ##
########################################################################################################################

# Prefix for the prompts in semantic dipole
face_prefix = "An ID photo of"
other_prefix = "A photo of"

########################################################################################################################
##                                                                                                                    ##
##                                            [ Semantic Dipoles Corpora ]                                            ##
##                                                                                                                    ##
########################################################################################################################
EXPRESSIONS_SHORT_DESCRIPTION = {
    'Surprise': "{} a surprised person.".format(face_prefix),
    'Disgust': "{} a disgusted person.".format(face_prefix),
    'Anger': "{} an angry person.".format(face_prefix),
    'Sadness': "{} a sad person.".format(face_prefix),
    'Happiness': "{} a happy person.".format(face_prefix),
    'Fear': "{} a fearful person.".format(face_prefix),
    'Neutral': "{} a person with neutral expression.".format(face_prefix)
}

EXPRESSIONS_LONG_DESCRIPTION = {
    'Surprise': "{} a surprised person with widen eyes, open mouth, and raised eyebrows with horizontal wrinkles on the forehead.".format(face_prefix),
    'Disgust': "{} a person with disgusted expression with wrinkled nose, raised upper lip, and pulled down eyebrows.".format(face_prefix),
    'Anger': "{} an angry person with eyebrows lowered and pulled closer together, squinted or raised eyelids, tightened lips, and tense jaw.".format(face_prefix),
    'Sadness': "{} a sad person with pulled down lip corners, upper eyelids drooped, eyes looking down, and inner corners of the eyebrows pulled up and together.".format(face_prefix),
    'Happiness': "{} a happy person with slightly squint eyes with wrinkles at the eyes corners and raised cheeks, wrinkles running from the sides of the nose to the corners of the mouth.".format(face_prefix),
    'Fear': "{} a fearful person with eyebrows pulled up and lowered eyelids, stretched mouth, and vertical wrinkles between the eyebrows.".format(face_prefix),
    'Neutral': "{} a person with neutral expression.".format(face_prefix)
}

SEMANTIC_DIPOLES_CORPORA = {
    'expressions6':
        [
            [EXPRESSIONS_LONG_DESCRIPTION['Sadness'], EXPRESSIONS_LONG_DESCRIPTION['Anger']],
            [EXPRESSIONS_LONG_DESCRIPTION['Fear'], EXPRESSIONS_LONG_DESCRIPTION['Happiness']],
            [EXPRESSIONS_LONG_DESCRIPTION['Disgust'], EXPRESSIONS_LONG_DESCRIPTION['Surprise']],
        ],
    'expressions6+attributes':
        [
            [EXPRESSIONS_LONG_DESCRIPTION['Sadness'], EXPRESSIONS_LONG_DESCRIPTION['Anger']],
            [EXPRESSIONS_LONG_DESCRIPTION['Fear'], EXPRESSIONS_LONG_DESCRIPTION['Happiness']],
            [EXPRESSIONS_LONG_DESCRIPTION['Disgust'], EXPRESSIONS_LONG_DESCRIPTION['Surprise']],
            ["{} an old person.".format(face_prefix),
             "{} a young person.".format(face_prefix)],
            ["{} a female.".format(face_prefix),
             "{} a male.".format(face_prefix)],
            ["{} a face with makeup.".format(face_prefix),
             "{} a face without makeup.".format(face_prefix)]
        ],
    'attributes':
        [
            ["{} an old person.".format(face_prefix),
             "{} a young person.".format(face_prefix)],
            ["{} a female.".format(face_prefix),
             "{} a male.".format(face_prefix)],
            ["{} a face with makeup.".format(face_prefix),
             "{} a face without makeup.".format(face_prefix)],
        ],
    'dogs':
        [
            ["{} a happy dog.".format(other_prefix),
             "{} a sad dog.".format(other_prefix)],
            ["{} a long haired dog.".format(other_prefix),
             "{} a short haired dog.".format(other_prefix)],
            ["{} a friendly dog.".format(other_prefix),
             "{} an aggressive dog.".format(other_prefix)],
            ["{} a dog with big eyes.".format(other_prefix),
             "{} a dog with small eyes.".format(other_prefix)]
        ],
    'cats':
        [
            ["{} a long haired cat.".format(other_prefix),
             "{} a short haired cat.".format(other_prefix)],
            ["{} a cute cat.".format(other_prefix),
             "{} an ugly cat.".format(other_prefix)],
            ["{} a cat with big ears.".format(other_prefix),
             "{} a cat with small ears.".format(other_prefix)]
        ],
    'cars':
        [
            ["Jeep", "Sports"],
            ["Modern", "From Sixties"],
        ]
}


########################################################################################################################
##                                                                                                                    ##
##                                                      [ FaRL ]                                                      ##
##                                                                                                                    ##
########################################################################################################################
# Choose pre-trained FaRL model (epoch 16 or 64)
FARL_EP = 64
FARL_PRETRAIN_MODEL = 'FaRL-Base-Patch16-LAIONFace20M-ep{}.pth'.format(FARL_EP)

FARL = ('https://www.dropbox.com/s/xxhmvo3q7avlcac/farl.tar?dl=1',
        '1d67cc6fd3cea9fdd7ec6af812a32e6b02374162d02137dd80827283d496b2d8')

########################################################################################################################
##                                                                                                                    ##
##                                                     [ SFD ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
SFD = ('https://www.dropbox.com/s/jssqpwyp4edp20o/sfd.tar?dl=1',
       '2bea5f1c10110e356eef3f4efd45169100b9c7704eb6e6abd309df58f34452d4')

########################################################################################################################
##                                                                                                                    ##
##                                                    [ ArcFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
ARCFACE = ('https://www.dropbox.com/s/idulblr8pdrmbq1/arcface.tar?dl=1',
           'edd5854cacd86c17a78a11f70ab8c49bceffefb90ee070754288fa7ceadcdfb2')

########################################################################################################################
##                                                                                                                    ##
##                                                   [ FairFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
FAIRFACE = ('https://www.dropbox.com/s/lqrydpw7nv27ass/fairface.tar?dl=1',
            '0e78ff8b79612e52e226461fb67f6cff43cef0959d1ab2b520acdcc9105d065e')

########################################################################################################################
##                                                                                                                    ##
##                                                    [ HopeNet ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
HOPENET = ('https://www.dropbox.com/s/rsw7gmo4gkqrbsv/hopenet.tar?dl=1',
           '8c9d67dd8f82ce3332c43b5fc407dc57674d1f16fbe7f0743e9ad57ede73e33f')

########################################################################################################################
##                                                                                                                    ##
##                                                 [ AU Detector ]                                                    ##
##                                                                                                                    ##
########################################################################################################################
AUDET = ('https://www.dropbox.com/s/jkkf1gda9o8ed47/au_detector.tar?dl=1',
         'dbdf18bf541de3c46769d712866bef38496b7528072850c28207747b2b2c101e')

########################################################################################################################
##                                                                                                                    ##
##                                              [ CelebA Attributes ]                                                 ##
##                                                                                                                    ##
########################################################################################################################
CELEBA_ATTRIBUTES = ('https://www.dropbox.com/s/bxbegherkpvgbw9/celeba_attributes.tar?dl=1',
                     '45276f2df865112c7488fe128d8c79527da252aad30fc541417b9961dfdd9bbc')


########################################################################################################################
##                                                                                                                    ##
##                                                     [ FER ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
FER = ('https://www.dropbox.com/s/1u6e7yvss56nx1n/fer.tar?dl=1',
       '94b1f8c23dfc5e626c1de0e76257f174463b8f2c371670036ce75f0923d4985d')

########################################################################################################################
##                                                                                                                    ##
##                                             [ GenForce GAN Generators ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GENFORCE = ('https://www.dropbox.com/s/3osul10173lbhut/genforce.tar?dl=1',
            'e0a250c8ed57935a3a60521d57db11f8aeaea6b791b8e072df27eba3c233f535')

GENFORCE_MODELS = {
    # ===[ ProgGAN ]===
    'pggan_celebahq1024': ('pggan_celebahq1024.pth', 1024),
    # ===[ StyleGAN2 ]===
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_ffhq512': ('stylegan2_ffhq512.pth', 512),
    'stylegan2_afhqcat512': ('stylegan2_afhqcat512.pth', 512),
    'stylegan2_afhqdog512': ('stylegan2_afhqdog512.pth', 512),
    'stylegan2_car512': ('stylegan2_car512.pth', 512),
}

STYLEGAN_LAYERS = {
    'stylegan2_ffhq1024': 18,
    'stylegan2_ffhq512': 16,
    'stylegan2_afhqcat512': 16,
    'stylegan2_afhqdog512': 16,
    'stylegan2_car512': 16,
    'stylegan2_church256': 14,
}

GAN_BASE_LATENT_DIM = {
    'stylegan2_ffhq1024': 512,
    'stylegan2_ffhq512': 512,
    'stylegan2_afhqcat512': 512,
    'stylegan2_afhqdog512': 512,
    'stylegan2_car512': 512,
    'stylegan2_church256': 512,
    'pggan_celebahq1024': 512
}

STYLEGAN2_STYLE_SPACE_TARGET_LAYERS = {
    'stylegan2_ffhq1024':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            # 'style08': 512,  # 'layer8'  : '64x64/Conv1'
            # 'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            # 'style10': 256,  # 'layer10' : '128x128/Conv1'
            # 'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            # 'style12': 128,  # 'layer12' : '256x256/Conv1'
            # 'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            # 'style14': 64,   # 'layer14' : '512x512/Conv1'
            # 'style15': 64,   # 'layer15' : '1024x1024/Conv0_up'
            # 'style16': 32    # 'layer16' : '1024x1024/Conv1'
        },
    'stylegan2_ffhq512':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            # 'style08': 512,  # 'layer8'  : '64x64/Conv1'
            # 'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            # 'style10': 256,  # 'layer10' : '128x128/Conv1'
            # 'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            # 'style12': 128,  # 'layer12' : '256x256/Conv1'
            # 'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            # 'style14': 64,   # 'layer14' : '512x512/Conv1'
        },
    'stylegan2_afhqcat512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            # 'style08': 512,
            # 'style09': 512,
            # 'style10': 256,
            # 'style11': 256,
            # 'style12': 128,
            # 'style13': 128,
            # 'style14': 64,
        },
    'stylegan2_afhqdog512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            # 'style08': 512,
            # 'style09': 512,
            # 'style10': 256,
            # 'style11': 256,
            # 'style12': 128,
            # 'style13': 128,
            # 'style14': 64,
        },
    'stylegan2_car512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            # 'style08': 512,
            # 'style09': 512,
            # 'style10': 256,
            # 'style11': 256,
            # 'style12': 128,
            # 'style13': 128,
            # 'style14': 64,
        },
    'stylegan2_church256':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            # 'style08': 512,
            # 'style09': 512,
            # 'style10': 256,
            # 'style11': 256,
            # 'style12': 128,
        }
}
