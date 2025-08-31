library(ggplot2)
library(gridExtra)
library(truncnorm)

## Define Path
path <- "./"

slim_model <- "onlyAM.slim"
totalpop <- c(1000)
REDS <- c(0.45, 0.1) # AFR, NAT
pops <- REDS

GEN <- 19 # generations
ITE <- 100 # runs with different seeds
LA0 <- 2860000000
LX0 <- 2990000000
scale <- c("1000")
LAstep <- 100

## Directory Setup
pop_name <- "REDS"
if (!dir.exists(paste0(path, pop_name))) {
    dir.create(paste0(path, pop_name))
    dir.create(paste0(path, pop_name, "/ancestries"))
    dir.create(paste0(path, pop_name, "/fragments"))
}

## Check for Previous Runs
AMcomb_pop_file <- paste0(path, "AM_combinations_onlyAM_scale_", scale, "_pop_REDS.txt")
lastrun <- 0
if (file.exists(AMcomb_pop_file)) {
    AMcomb_pop <- read.table(AMcomb_pop_file)
    lastrun <- max(AMcomb_pop$V1)
}

## Remove Old Run Script
system(paste0("rm -f ", path, "runrunfile_onlyAM_scale_100000.sh"))

## Setup Simulation
for (SCALE in scale) {
    for (TOTALPOP in totalpop) {
        RATPOP1 <- pops[1] # AFR
        RATPOP2 <- pops[2] # NAT
        POP1 <- round(TOTALPOP * RATPOP1)
        POP2 <- round(TOTALPOP * RATPOP2)
        POP3 <- round(TOTALPOP - POP1 - POP2)

        # Define Input/Output Files
        output_jobfile <- paste0(path, pop_name, "/Slim_output_onlyAM_GEN_", GEN, "_POP1_", RATPOP1, "_POP2_", RATPOP2, "_totalpop_", TOTALPOP, "_pop_REDS_scale_", SCALE, "_run_$1.txt")
        fragsfile <- paste0(path, pop_name, "/fragments/Frags_onlyAM_GEN_", GEN, "_POP1_", RATPOP1, "_POP2_", RATPOP2, "_totalpop_", TOTALPOP, "_pop_REDS_scale_", SCALE, "_run_$1.txt")
        ancfile <- paste0(path, pop_name, "/ancestries/Ancestries_onlyAM_GEN_", GEN, "_POP1_", RATPOP1, "_POP2_", RATPOP2, "_totalpop_", TOTALPOP, "_pop_REDS_scale_", SCALE, "_run_$1.txt")

        # SLiM Command
        command_runjobfile <- paste0("/opt/anaconda/bin/slim -d POP=1",
                                    " -d SB1_t1=0.0",
                                    " -d SB2_t1=0.0",
                                    " -d GEN=", GEN,
                                    " -d RUN=$1",
                                    " -d LA0=", LA0,
                                    " -d LX0=", LX0,
                                    " -d SCALE=", SCALE,
                                    " -d LAstep=", LAstep,
                                    " -d totalpop=", TOTALPOP,
                                    " -d pop1_ratio=", RATPOP1,
                                    " -d pop2_ratio=", RATPOP2,
                                    " -d prop_pop1_t1=1.0",
                                    " -d prop_pop2_t1=1.0",
                                    " -d prop_pop3_t1=1.0",
                                    " ", slim_model, " > ", output_jobfile)

        # Output Splitting Commands
        command_split_output_1 <- paste0("linefrag=$(grep -n ^###Frag ", output_jobfile, " | sed -r 's/([0-9]+):.*/\\1/g')")
        command_split_output_2 <- paste0("tail -n+$(($linefrag+1)) ", output_jobfile, " > ", fragsfile)
        command_split_output_3 <- paste0("lineanc=$(grep -n ^###Ancestry ", output_jobfile, " | sed -r 's/([0-9]+):.*/\\1/g')")
        command_split_output_4 <- paste0("tail -n+$(($lineanc+1)) ", output_jobfile, " | head -n $(($linefrag-$lineanc-1)) > ", ancfile)

        # Create Shell Script for Parallel Execution
        runjobfile <- paste0(path, pop_name, "/run_job_onlyAM_GEN_", GEN, "_POP1_", RATPOP1, "_POP2_", RATPOP2, "_totalpop_", TOTALPOP, "_pop_REDS_scale_", SCALE, ".sh")
        write("#!/bin/bash", runjobfile)
        write(paste0("cd ", path), runjobfile, append=TRUE)
        write(paste0(command_runjobfile, " && ", command_split_output_1, " && ", command_split_output_2, " && ", command_split_output_3, " && ", command_split_output_4), runjobfile, append=TRUE)

        system(paste0("chmod 770 ", runjobfile))

        # Create Parallel Run Script
        parallel_command <- paste0("parallel -j100 -- ", runjobfile, " {} ::: $(seq ", lastrun + 1, " ", lastrun + ITE, ")")
        write("#!/bin/bash", paste0(path, "runrunfile_onlyAM_scale_100000.sh"), append=TRUE)
        write(parallel_command, paste0(path, "runrunfile_onlyAM_scale_100000.sh"), append=TRUE)
        system(paste0("chmod 770 ", path, "runrunfile_onlyAM_scale_100000.sh"))

        # Execute Parallel Run Script
        system(paste0("bash ", path, "runrunfile_onlyAM_scale_100000.sh"))
    }
}
