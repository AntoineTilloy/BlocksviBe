# Builds b3d_setup (blocks_3d's M-matrix and h_infinity stages, see b3d_setup.cxx) from a blocks_3d
# source tree (https://gitlab.com/bootstrapcollaboration/blocks_3d, tested at 58c9717).
#   make B3D=<blocks_3d checkout> ENV=<prefix with include/ and lib/ for boost, gmp, eigen3, fmt>
# e.g. ENV = a conda/micromamba env with boost-cpp gmp eigen fmt, or /usr (Debian: libboost-dev
# libgmp-dev libeigen3-dev libfmt-dev).  Output: ../build/b3d_setup (where blocksvibe looks first).
B3D ?= $(HOME)/Projects/sdpb_gpu/ext/src_b3d/blocks_3d
ENV ?= $(HOME)/Projects/sdpb_gpu/ext/mamba/b3denv
CXX ?= g++
OUT ?= ../build
CXXFLAGS = -O3 -std=c++17 -I$(B3D)/src -I$(ENV)/include -I$(ENV)/include/eigen3
T = src/lambda_log_r_tables
D = $(T)/lambda_log_r_deriv/conformal_block_derivatives
SRCS = src/compute_pseudo_parity_types.cxx src/so3_tensor_parity.cxx $(T)/Clebsch_Gordan.cxx \
  $(T)/M_Matrix/calculate/M_Matrix.cxx $(T)/M_Matrix/calculate/M/M.cxx $(T)/M_Matrix/calculate/M/d_normalized.cxx \
  $(T)/Pole_Types.cxx $(T)/F_Q_Basis_Derivative.cxx $(T)/Wigner_Cache/Wigner_Cache.cxx \
  $(T)/Wigner_Cache/wjm_matrix.cxx $(T)/Wigner_Cache/coefficient.cxx \
  $(T)/f_derivatives/F_SO3_Basis_Derivative.cxx $(T)/f_derivatives/f_W_conversion_matrix.cxx \
  $(T)/f_derivatives/f_q_basis_derivative.cxx $(D)/fill_binomials.cxx $(D)/h_infinity_double_series.cxx \
  $(D)/two_j_values.cxx $(D)/h_infinity/h_infinity.cxx $(D)/h_infinity/full_derivative.cxx \
  $(D)/clebsch_gordan.cxx $(D)/h_infinity/prefactor_derivatives/prefactor_derivatives.cxx
OBJS = $(patsubst %.cxx,$(OUT)/obj/%.o,$(SRCS))

$(OUT)/b3d_setup: $(OUT)/obj/b3d_setup.o $(OBJS)
	$(CXX) -o $@ $^ -L$(ENV)/lib -Wl,-rpath,$(ENV)/lib -lgmpxx -lgmp -lpthread
$(OUT)/obj/b3d_setup.o: b3d_setup.cxx
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(B3D)/src -c $< -o $@
$(OUT)/obj/%.o: $(B3D)/%.cxx
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
clean:
	rm -rf $(OUT)/obj $(OUT)/b3d_setup
