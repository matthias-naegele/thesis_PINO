!================================================================================
!
!    BHAC (The Black Hole Accretion Code) solves the equations of
!    general relativistic magnetohydrodynamics and other hyperbolic systems
!    in curved spacetimes.
!
!    Copyright (C) 2019 Oliver Porth, Hector Olivares, Yosuke Mizuno, Ziri Younsi,
!    Luciano Rezzolla, Elias Most, Bart Ripperda and Fabio Bacchini
!
!    This file is part of BHAC.
!
!    BHAC is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    BHAC is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    along with BHAC.  If not, see <https://www.gnu.org/licenses/>.
!
!================================================================================

module mod_indices
   implicit none

   !AMR specific parameters
   !
   ! ngridshi:  maximum number of grids
   ! nlevelshi: maximum number of levels in grid refinement

   integer, parameter :: ngridshi = 1800000
   integer, parameter :: nlevelshi = 13

   ! number of interleaving sending buffers for ghostcells
   integer, parameter :: npwbuf=2

   integer, save :: ixM^LL

   integer, dimension(nlevelshi), save :: ng^D
   double precision, dimension(nlevelshi), save :: dg^D

   logical, save :: slab, covariant

   integer, save :: npe, mype, icomm
   integer, save :: log_fh
   integer, save :: type_block, type_coarse_block, type_sub_block(2^D&)
   integer, save :: type_block_io, size_block_io
{#IFDEF TRANSFORMW
   integer, save :: type_block_io_tf, size_block_io_tf}
   integer, save :: type_subblock_io, type_subblock_x_io   ! Center variables
   integer, save :: type_subblockC_io, type_subblockC_x_io ! Corner variables
   integer, save :: type_block_xc_io,type_block_xcc_io
   integer, save :: type_block_wc_io,type_block_wcc_io
!   integer, save :: itag, ierrmpi
   integer, asynchronous :: itag, ierrmpi

   integer, save :: irecv, isend
{#IFDEF STAGGERED
   integer, save :: itag_stg
   integer, save :: itag_cc, ierrmpi_cc
   integer, save :: irecv_cc, isend_cc
}
   integer, dimension(:), allocatable, save :: recvrequest, sendrequest
   integer, dimension(:,:), allocatable, save :: recvstatus, sendstatus
{#IFDEF STAGGERED   
   integer, dimension(:), allocatable, save :: recvrequest_stg, sendrequest_stg
   integer, dimension(:,:), allocatable, save :: recvstatus_stg, sendstatus_stg
   integer, save  :: type_coarse_block_stg(^ND,2^D&), type_sub_block_stg(^ND,2^D&)
   integer, save  :: type_block_staggered_io, size_block_stg_io
}
   integer, save :: snapshot, snapshotnext, slice, slicenext, collapseNext, icollapse, ishell, shellNext

   logical, allocatable, dimension(:^D&), save :: patchfalse

   logical, save :: B0field
   double precision, save :: Bdip, Bquad, Boct, Busr

end module mod_indices
