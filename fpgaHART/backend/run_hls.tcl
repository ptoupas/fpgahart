set script_path [ file dirname [ file normalize [ info script ] ] ]
set partition_name [file tail $script_path]
set partition_name [string tolower $partition_name]

set project_name $partition_name
append project_name "_vitis_prj"

set top_level $partition_name
append top_level "_top"

# Create a project
open_project $project_name -reset

set src_files [glob src/*.cpp]
set tb_files [glob tb/*.cpp]

foreach s_f $src_files {
	add_files $s_f
}

foreach t_f $tb_files {
	add_files -tb $t_f -cflags "-Itb -Idata -Isrc -I../../include -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
}

set_top $top_level

# ########################################################
# Create a solution
open_solution -reset solution1 -flow_target vivado
# Define technology and clock rate
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 7 -name default
config_dataflow -default_channel fifo

# Set variable to select which steps to execute
set hls_exec 2

#csim_design -clean -O

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design
	
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	
	cosim_design -O -enable_dataflow_profiling
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	
	cosim_design -O -enable_dataflow_profiling

	export_design -format ip_catalog
} else {
	# Default is to exit after setup
	csynth_design
}

exit
