
import BIDSification_eyetrackingData as BIDS


path_oldData = '/Volumes/work/brainets/oueld.h/Contextual Learning/data' # Path of the data directory to BIDSified
path_newData = '/Volumes/work/brainets/oueld.h/Contextual Learning/dataBIDS' # Path of the new BIDS data directory


# Creating the necessary files for the BIDSification
process = BIDS.StandardisationProcess(dirpath=path_oldData)
#------------------------------------------------------------------------------
process.create_settingsFile()
#------------------------------------------------------------------------------
process.create_dataset_description()
#------------------------------------------------------------------------------
process.create_settingsEvents('infoFiles.tsv')

|%%--%%| <IjGooByzHu|zj7rJ5CoZ8>





# Name of the file describing the dataset
datasetdescriptionfilename = 'dataset_description.json'
# Name of the file containing the information on the files to be BIDSified
infofilesname = 'infoFiles.tsv'
# Name of the file containing the information on the files to be BIDSified
settingsfilename = 'settings.json'
# Name of the file containing the events settings in the BIDSified data
settingsEventsfilename = 'settingsEvents.json'


eyetracktype = 'Eyelink' # Name of the type of eyetrackeur used
dataformat = '.asc' # Data format

# List of events to be extracted from the trials
saved_events = {"FixOn": {"Description": "appearance of the fixation point"},
                "FixOff": {"Description": "disappearance of the fixation point"},
                "TargetOn": {"Description": "appearance of the moving target"},
                "TargetOff": {"Description": "disappearance of the moving target"}}

StartMessage= 'color'# Message marking the start of the trial
EndMessage= 'TargetOff' # Message marking the end of the trial



#------------------------------------------------------------------------------
# to apply the function:
#------------------------------------------------------------------------------
BIDS.DataStandardisation(path_oldData=path_oldData,
                         path_newData=path_newData,
                         datasetdescriptionfilename=datasetdescriptionfilename,
                         infofilesname=infofilesname,
                         settingsfilename=settingsfilename,
                         settingsEventsfilename=settingsEventsfilename,
                         eyetracktype=eyetracktype,
                         dataformat=dataformat,
                         saved_events=saved_events,
                         StartMessage=StartMessage,
                         EndMessage=EndMessage);


#|%%--%%| <deLSqMsnQS|7ijilYl3cB>

#|%%--%%| <7ijilYl3cB|EX6hjLnbyP>

#|%%--%%| <EX6hjLnbyP|ETODeUt3so>


process = BIDS.StandardisationProcess(dirpath=path_oldData)

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
dataformat = '.asc' # Data format

#------------------------------------------------------------------------------
# to apply the function:
#------------------------------------------------------------------------------
process.create_infoFiles(dataformat=dataformat)

#|%%--%%| <ETODeUt3so|AHrpxf65Zu>
import ANEMO
rawDataDirPath='/Volumes/work/brainets/oueld.h/Contextual Learning/data'
processedDataDirPath='processedData'
ANEMO.init(rawDataDirPath,processedDataDirPath,sub='01')
#|%%--%%| <AHrpxf65Zu|xUoVj2egvC>


# A new folder has just been created
print('RawDatadirpath directory tree:')
ANEMO.Data.dirtree(rawDataDirPath)

#|%%--%%| <xUoVj2egvC|VspYBKQhaQ>


print('Datadirpath directory tree:')
ANEMO.Data.dirtree(processedDataDirPath)

