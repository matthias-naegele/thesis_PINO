############################################################################
#
#                               BHAC Makefile
#
############################################################################
#
# For the first time read carefully everything and do selections.
# Later you will probably need to change the compiler flags only.
# You may delete compiler flags for other machines once everything works.
#
# Select compiling options by adding and removing the leading # characters.
#
############################################################################

# The Makefile should always use the sh shell.
SHELL         = /bin/sh
# The preprocessed *.f90 files will be put into a separate subfolder
FOBJECT_DIR   = F90sources
COBJECT_DIR   = Csources
OBJ_DIR       = Build

ARCH          = gfortran10_hdf5.defs
SRC           = $(BHAC_DIR)/src

include $(BHAC_DIR)/arch/$(ARCH)

############## setup.pl: #######################
PHYSICS       = rrmhd

VPATH         =  $(WORK_DIR):$(SRC)/$(PHYSICS):$(SRC)
INCLUDE_DIRS  += .
################################################

############### VACPreProcessor ################
#Collect data for preprocessor set by setup.pl:
ndim=2
ndir=3
phi=2
z=3
nf=0
eos=default
coord=cart
# SETVAC READS UP TO THIS POINT

PPFLAGS  = -d=$(ndim)$(ndir) -phi=$(phi)
PPFLAGS += -z=$(z) -p=$(PHYSICS) -nf=$(nf)
PPFLAGS += -eos=$(eos)
PPFLAGS += -coord=$(coord)

# For the compilation rules:
VACPP =$(BHAC_DIR)/src/vacpp.pl #

############################################################################
# EXTENSION NAMES FOR SOURCE AND OBJECT FILES
#
# $F is the suffix for the generic AMRVAC precompiled Fortran files
# $O is the suffix for the object files, usually .o
F=.f90
C=.c
O=.o
I=.i

##############################################################################
######                                                                 #######
######   YOU PROBABLY DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE   #######
######   BUT IT MAY BE WORTHWHILE TO READ IT                           #######
######                                                                 #######
##############################################################################

# Default target
bhac:


################### Include the files in the src-dir #######################
-include $(SRC)/makefile

################### Include the geometry sources in the geometry dir #######
VPATH += :$(SRC)/geometry
-include $(SRC)/geometry/makefile

################### Include the IO-related objects in amrvacio dir #########
VPATH += :$(SRC)/amrvacio
-include $(SRC)/amrvacio/makefile

################### Include additional objects in physics dir ##############
-include $(SRC)/$(PHYSICS)/makefile

################### Include additional modules in modules dir ##############
VPATH += :$(SRC)/modules
-include $(SRC)/modules/makefile

################### Include the particle modules ###########################
VPATH += :$(SRC)/particles
-include $(SRC)/particles/makefile

################### Include additional files in local dir     ##############
-include $(WORK_DIR)/localmakefile

################### Rules in Build directory ####### #######################
ifeq (1,$(RECURSED))
############################################################################

.SUFFIXES: .t $F $O $C

########### C ############################

COBJECT_DIR := $(WORK_DIR)/$(COBJECT_DIR)

COBJECTS_IN_FOLDER = $(addprefix $(COBJECT_DIR)/, $(COBJECTS))

$(COBJECT_DIR)/%$(C) :%$(C)
	@mkdir -p $(@D)
	cp $< $@

# General C compilation rule
%$(O) :$(COBJECT_DIR)/%$(C)
	$(CC) $(CCFLAGS) $<

########### FORTRAN ######################

FOBJECT_DIR := $(WORK_DIR)/$(FOBJECT_DIR)

FOBJECTS_IN_FOLDER = $(addprefix $(FOBJECT_DIR)/, $(FOBJECTS))
INCLUDES_IN_FOLDER = $(addprefix $(FOBJECT_DIR)/, $(INCLUDES))


PREFOR= #
PREPROC= $(VACPP_)


$(FOBJECTS_IN_FOLDER)	: $(PREPROC)
$(OBJECTS)	        : $(INCLUDES_IN_FOLDER)


# General precompilation rule
$(FOBJECT_DIR)/%$(F) :%.t
	@mkdir -p $(@D)	
	$(VACPP) $(PPFLAGS) $< $(PREFOR) > $@

# General Fortran compilation rule
%$(O) :$(FOBJECT_DIR)/%$(F)
	$(F90) $(FFLAGS) $(F90FLAGS) $(addprefix -I, $(INCLUDE_DIRS)) $<

# Will be called to update dependent objects
objects: $(OBJECTS)

# Link the code
bhac: $(OBJECTS)
	$(LINK) $(OBJECTS) $(LIBS) -o $(WORK_DIR)/$@


############################################################################
else
############################################################################
OBJECTS_IN_FOLDER  = $(addprefix $(OBJ_DIR)/, $(OBJECTS))

bhac: $(OBJECTS_IN_FOLDER)
	+$(MAKE) -C $(OBJ_DIR) -f $(CURDIR)/makefile WORK_DIR=$(CURDIR) \
	RECURSED=1 bhac

# This is needed to re-make when the source changes
$(OBJECTS_IN_FOLDER): $(OBJ_DIR)
	@+$(MAKE) -C $(OBJ_DIR) -f $(CURDIR)/makefile WORK_DIR=$(CURDIR) \
	RECURSED=1 objects

# Build folder
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)


#########################################################################
#  Archive the setup
#########################################################################
ARCHIVE = $(shell date "+Setup-%Y-%m-%d")
archive: 
	@echo === Archiving into $(ARCHIVE).tgz   ===
	@git -C $(BHAC_DIR) log -1 > $(ARCHIVE)-gitversion.txt
	@git -C $(BHAC_DIR) status >> $(ARCHIVE)-gitversion.txt
	@tar -czvf $(ARCHIVE).tgz *.t *.par *makefile definitions.h $(ARCHIVE)-gitversion.txt
	@echo === Created $(ARCHIVE).tgz ===


#########################################################################
#  Make clean
#########################################################################
clean:
	/bin/rm -rf $(OBJ_DIR) $(FOBJECT_DIR) bhac

oldclean:
	/bin/rm -f bhac *.f90 *.o *.mod



endif
