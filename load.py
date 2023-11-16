datasets = [
    'Adiac',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'Car',
    'CBF',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxTW',
    'Earthquakes',
    'ECG200',
    # 'ECG5000',
    'ECGFiveDays',
    # 'ElectricDevices',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    # 'FordA',
    # 'FordB',
    'GunPoint',
    'Ham',
    # 'HandOutlines',
    # 'Haptics',
    'Herring',
    # 'InlineSkate',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    # 'Mallat',
    'Meat',
    'MedicalImages',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',    
    'MiddlePhalanxTW',
    'MoteStrain',
    # 'NonInvasiveFatalECGThorax1',
    # 'NonInvasiveFatalECGThorax2',
    'OliveOil',
    'OSULeaf',
    'PhalangesOutlinesCorrect',
    # 'Phoneme',
    'Plane',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    # 'ScreenType',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    # 'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'UWaveGestureLibraryAll',
    # 'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    # 'WormsTwoClass',
    # 'Yoga'
]

metrics = ['euclidean', 'dtw']  # , 'softdtw']
header = [
    'points',
    'adj_rand_mean',
    'adj_rand_std',
    'adj_mut_mean',
    'adj_mut_std',
    'run_time', 
    'run_time_base']

images = [('adj_rand_mean', 'adj_rand_std')]

# ('adj_mut_mean', 'adj_mut_std')]
baseResults = 'results/'
baseAnalytics = 'analytics/'

a = ['$0$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$']
x = [1, 2, 3, 4, 5]
ARI = (1, 2)
AMI = (3, 4)
RUNTIME = (5, 6)
ARI_LABEL = 'ARI'
AMI_LABEL = 'AMI'

ADJ_MUT = 'adj_mutual_info'
ADJ_RAND = 'adj_rand_ind'
