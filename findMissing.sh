#!/bin/bash
# Define the list of strings
# stringList=("Adiac" "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "Car" "CBF" "ChlorineConcentration" "CinCECGTorso" "Coffee" "Computers" "CricketX" "CricketY" "CricketZ" "DiatomSizeReduction" "DistalPhalanxOutlineCorrect" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "Earthquakes" "ECG200" "ECG5000" "ECGFiveDays" "ElectricDevices" "FaceAll" "FaceFour" "FacesUCR" "FiftyWords" "Fish" "FordA" "FordB" "GunPoint" "Ham" "HandOutlines" "Haptics" "Herring" "InlineSkate" "InsectWingbeatSound" "ItalyPowerDemand" "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat" "MedicalImages" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "MoteStrain" "NonInvasiveFatalECGThorax1" "NonInvasiveFatalECGThorax2" "OliveOil" "OSULeaf" "PhalangesOutlinesCorrect" "Phoneme" "Plane" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW" "RefrigerationDevices" "ScreenType" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry" "SwedishLeaf" "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace" "TwoLeadECG" "TwoPatterns" "UWaveGestureLibraryX" "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "UWaveGestureLibraryAll" "Wafer" "Wine" "WordSynonyms" "Worms" "WormsTwoClass" "Yoga")
stringList=(
  "Adiac"
  "ArrowHead"
  "Beef"
  "BeetleFly"
  "BirdChicken"
  "Car"
  "CBF"
  "ChlorineConcentration"
  "CinCECGTorso"
  "Coffee"
  "Computers"
  "CricketX"
  "CricketY"
  "CricketZ"
  "DiatomSizeReduction"
  "DistalPhalanxOutlineCorrect"
  "DistalPhalanxOutlineAgeGroup"
  "DistalPhalanxTW"
  "Earthquakes"
  "ECG200"
  "ECG5000"
  "ECGFiveDays"
  "ElectricDevices"
  "FaceAll"
  "FaceFour"
  "FacesUCR"
  "FiftyWords"
  "Fish"
  "FordA"
  "FordB"
  "GunPoint"
  "Ham"
  "HandOutlines"
  "Haptics"
  "Herring"
  "InlineSkate"
  "InsectWingbeatSound"
  "ItalyPowerDemand"
  "LargeKitchenAppliances"
  "Lightning2"
  "Lightning7"
  "Mallat"
  "Meat"
  "MedicalImages"
  "MiddlePhalanxOutlineCorrect"
  "MiddlePhalanxOutlineAgeGroup"
  "MiddlePhalanxTW"
  "MoteStrain"
  "NonInvasiveFatalECGThorax1"
  "NonInvasiveFatalECGThorax2"
  "OliveOil"
  "OSULeaf"
  "PhalangesOutlinesCorrect"
  "Phoneme"
  "Plane"
  "ProximalPhalanxOutlineCorrect"
  "ProximalPhalanxOutlineAgeGroup"
  "ProximalPhalanxTW"
  "RefrigerationDevices"
  "ScreenType"
  "ShapeletSim"
  "ShapesAll"
  "SmallKitchenAppliances"
  "SonyAIBORobotSurface1"
  "SonyAIBORobotSurface2"
  "StarLightCurves"
  "Strawberry"
  "SwedishLeaf"
  "Symbols"
  "SyntheticControl"
  "ToeSegmentation1"
  "ToeSegmentation2"
  "Trace"
  "TwoLeadECG"
  "TwoPatterns"
  "UWaveGestureLibraryX"
  "UWaveGestureLibraryY"
  "UWaveGestureLibraryZ"
  "UWaveGestureLibraryAll"
  "Wafer"
  "Wine"
  "WordSynonyms"
  "Worms"
  "WormsTwoClass"
  "Yoga"
)
# Specify the directory where the files are located
directory="./defaultResults"

# Loop through each string in the list
for string in "${stringList[@]}"
do
    # Check if the DTW file does not exist
    dtwFile="$directory/${string}_dtw.csv"
    if [ ! -f "$dtwFile" ]; then
        echo "1, dtw, $string"
    fi

    # Check if the DTW preprocessed file does not exist
    dtwPreprocessedFile="$directory/${string}_dtw_preprocessed.csv"
    if [ ! -f "$dtwPreprocessedFile" ]; then
        echo "0, dtw, $string"
    fi

    # Check if the Euclidean file does not exist
    euclideanFile="$directory/${string}_euclidean.csv"
    if [ ! -f "$euclideanFile" ]; then
        echo "1, euclidean, $string"
    fi

    # Check if the Euclidean preprocessed file does not exist
    euclideanPreprocessedFile="$directory/${string}_euclidean_preprocessed.csv"
    if [ ! -f "$euclideanPreprocessedFile" ]; then
        echo "0, euclidean, $string"
    fi
done
