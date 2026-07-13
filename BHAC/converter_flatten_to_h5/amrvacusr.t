!=============================================================================
! amrvacusr.t.srmhdOT

! INCLUDE:amrvacnul/specialini.t
! INCLUDE:amrvacnul/speciallog.t
!INCLUDE:amrvacmodules/handle_particles.t
!INCLUDE:amrvacmodules/integrate_particles.t
INCLUDE:amrvacnul/specialbound.t
INCLUDE:amrvacnul/specialsource.t
INCLUDE:amrvacnul/specialimpl.t
INCLUDE:amrvacnul/usrflags.t
INCLUDE:amrvacnul/correctaux_usr.t
!=============================================================================
subroutine initglobaldata_usr
  use mod_amrvacdef
  implicit none
  double precision :: eta0
  integer :: ios, u

  namelist /usrlist/ eta0

  ! defaults if usrlist missing
  eta0 = 0.0d0

  u = 99
  open(unit=u, file='amrvac.par', status='old', action='read')
  read(u, nml=usrlist, iostat=ios)
  close(u)

  if (mype == 0 .and. ios /= 0) then
    write(*,*) 'Warning: could not read &usrlist; using eta0=', eta0
  end if

  eqpar(gamma_) = 4.0d0/3.0d0
  eqpar(eta_)   = eta0
end subroutine initglobaldata_usr

!=============================================================================
subroutine initonegrid_usr(ixI^L,ixO^L,s)

! initialize one grid

use mod_amrvacdef

integer, intent(in) :: ixI^L,ixO^L
!double precision, intent(in) :: x(ixG^S,1:ndim)
!double precision, intent(inout) :: w(ixG^S,1:nw)
type(state)                  :: s
double precision:: rho0,p0,vmax

logical, save :: first=.true.

{#IFDEF STAGGERED
integer :: ixGs^L
}

double precision :: nan
nan=0.0d0
nan=nan/nan

!----------------------------------------------------------------------------
{#IFDEF STAGGERED
! Limit indices for staggered variables
ixGsmin^D=s%ws%ixGmin^D;
ixGsmax^D=s%ws%ixGmax^D;
}

associate(x=>s%x%x,w=>s%w%w{#IFDEF STAGGERED ,ws=>s%ws%w})

!w=nan
  
rho0=one
p0=10.0d0
vmax=0.99d0

vmax=vmax/dsqrt(two)

w(ixO^S,rho_) =  rho0
w(ixO^S,v1_)  = -vmax*sin(x(ixO^S,2))
w(ixO^S,v2_)  =  vmax*sin(x(ixO^S,1))
w(ixO^S,v3_)  =  0.0d0
w(ixO^S,pp_)  =  p0


w(ixO^S,b1_) =-sin(x(ixO^S,2))
w(ixO^S,b2_) = sin(two*x(ixO^S,1))
w(ixO^S,b3_) = 0.0d0

{#IFNDEF STAGGERED
call b_from_vectorpotential(ixI^L,ixO^L,w,x)
}{#IFDEF STAGGERED
call b_from_vectorpotential(s%ws%ixG^L,ixI^L,ixO^L,ws,x)
call faces2centers(ixO^L,s)
}


{#IFDEF EPSINF
w(ixO^S,epsinf_)=one
w(ixO^S,rho0_)=one
w(ixO^S,rho1_)=one
w(ixO^S,n_)=one
w(ixO^S,n0_)=one
}

{#IFDEF GLM
w(ixO^S,psi_) = zero
}

{#IFDEF ENTROPY
  w(ixO^S, s_) = w(ixO^S, pp_) / ( w(ixO^S, rho_) ** eqpar(gamma_) )
}
 
w(ixO^S,lfac_)=one/dsqrt(one-({^C&w(ixO^S,v^C_)**2+}))
if(useprimitiveRel)then
   {^C&w(ixO^S,u^C_)=w(ixO^S,lfac_)*w(ixO^S,v^C_)\}
endif

w(ixO^S,e1_)  = (w(ixO^S,b2_)*w(ixO^S,u3_) - w(ixO^S,b3_)*w(ixO^S,u2_))/w(ixO^S,lfac_)
w(ixO^S,e2_)  = (w(ixO^S,b3_)*w(ixO^S,u1_) - w(ixO^S,b1_)*w(ixO^S,u3_))/w(ixO^S,lfac_)
w(ixO^S,e3_)  = (w(ixO^S,b1_)*w(ixO^S,u2_) - w(ixO^S,b2_)*w(ixO^S,u1_))/w(ixO^S,lfac_)

call conserve(ixI^L,ixO^L,w,x,patchfalse)

if(first.and.mype.eq.0)then
      write(*,*)'Doing 2D resistive SRMHD, Orszag Tang problem'
      write(*,*)'rho - p - gamma - primRel?:',rho0,p0,eqpar(gamma_),useprimitiveRel
      first=.false.
endif

end associate

return
end subroutine initonegrid_usr
!=============================================================================
subroutine initvecpot_usr(ixI^L, ixC^L, xC, A, idir)

  ! initialize the vectorpotential on the corners
  ! used by b_from_vectorpotential()

  use mod_amrvacdef

  integer, intent(in)                :: ixI^L, ixC^L,idir
  double precision, intent(in)       :: xC(ixI^S,1:ndim)
  double precision, intent(out)      :: A(ixI^S)
  ! .. local ..
  double precision                   :: bfactor
  !-----------------------------------------------------------------------------

  bfactor = 1.0d0
  if (idir.eq.3) then
    A(ixC^S) = bfactor*(half*cos(two*xC(ixC^S,1)) + cos(xC(ixC^S,2)))
  else
    A(ixC^S) = 0.0d0
  end if

end subroutine initvecpot_usr
!=============================================================================
subroutine specialvar_output(ixI^L,ixO^L,nwmax,w,s,normconv)

! this subroutine can be used in convert, to add auxiliary variables to the
! converted output file, for further analysis using tecplot, paraview, ....
! these auxiliary values need to be stored in the nw+1:nw+nwauxio slots
!
! the array normconv can be filled in the (nw+1:nw+nwauxio) range with 
! corresponding normalization values (default value 1)

use mod_amrvacdef

integer, intent(in)                :: ixI^L,ixO^L,nwmax
double precision                   :: w(ixI^S,nwmax)
type(state)                        :: s
double precision                   :: normconv(0:nwmax)
! .. local ..
double precision,dimension(ixI^S,1:ndir) :: current
!-----------------------------------------------------------------------------
associate(x=>s%x%x{#IFDEF STAGGERED ,ws=>s%ws%w})

          if (nwmax-nw .gt. 0) then

          call getcurrent(ixI^L,ixO^L,w(ixI^S,1:nw),x,current,.false.)
             w(ixO^S,nw+1) = current(ixO^S,3)
          end if

          if (nwmax-nw .gt. 1) then
             {#IFDEF STAGGERED
             call div_staggered(ixO^L,s,w(ixO^S,nw+2))
             }{#IFNDEF STAGGERED
          ! Reduce output array size, +1 was added for eventual pointdata output
             call get_divb(ixI^L,ixO^L^LSUB1,w(ixI^S,1:nw),w(ixI^S,nw+2))
             }
          end if
          if (nwmax-nw .gt. 2) then
             w(ixO^S,nw+3) = eqpar(eta_)
          end if

    end associate

end subroutine specialvar_output
!=============================================================================
subroutine specialvarnames_output

! newly added variables need to be concatenated with the varnames/primnames string

use mod_amrvacdef
!-----------------------------------------------------------------------------

primnames= TRIM(primnames)//' '//'jz'
wnames=TRIM(wnames)//' '//'jz'

primnames= TRIM(primnames)//' '//'divB'
wnames=TRIM(wnames)//' '//'divB'

primnames = TRIM(primnames)//' '//'eta'
wnames    = TRIM(wnames)//' '//'eta'


end subroutine specialvarnames_output
!=============================================================================
subroutine printlog_special

use mod_amrvacdef
!-----------------------------------------------------------------------------

call mpistop("special log file undefined")

end subroutine printlog_special
!=============================================================================
subroutine process_grid_usr(igrid,level,ixI^L,ixO^L,qt,w,x)

! this subroutine is ONLY to be used for computing auxiliary variables
! which happen to be non-local (like div v), and are in no way used for
! flux computations. As auxiliaries, they are also not advanced

use mod_amrvacdef

integer, intent(in):: igrid,level,ixI^L,ixO^L
double precision, intent(in):: qt,x(ixI^S,1:ndim)
double precision, intent(inout):: w(ixI^S,1:nw)
!-----------------------------------------------------------------------------

end subroutine process_grid_usr
!=============================================================================
! amrvacusr.t.srmhdOT
!=============================================================================
