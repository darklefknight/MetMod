module baromod
    implicit none
    !
    !     module baromod contains all global parameters/variables
    !     (i.e. which might be needed in  more that one subroutine)
    !     to be included be *use baromod*
    !
    !     a) constants:
    !
    integer, parameter :: NX = 64         ! x dimension
    integer, parameter :: NY = 32         ! y dimension
    real, parameter :: radea = 6.371E6    ! radius of the earth [m]
    real, parameter :: omega = 0.00007292 ! angular velocity [1/s]
    !
    integer :: nrun = 100        !  no timesteps to be computed
    integer :: nout = 1        !  output interval
    !
    real :: rlat = 50.          ! central latitude of the channel
    real :: xchannel = 360.         ! channel (x-) length [dec]
    real :: ychannel = 40.          ! channel (y-) width  [dec]
    real :: delt = 1200.        ! time step [s]
    !
    !     b) variables
    !
    integer :: nstep = 0        ! time step counter
    integer :: nwout = 0        ! output counter
    !
    real :: s(0 : NX + 1, 0 : NY + 1) = 0. ! streamfct. [m**2/2]
    real :: sm(0 : NX + 1, 0 : NY + 1) = 0. ! streamfct. (old timestep) [m**2/s]
    real :: dsdt(0 : NX + 1, 0 : NY + 1) = 0. ! streamfct tendnecy (ds/dt) [m**2/s**2]
    real :: vo(0 : NX + 1, 0 : NY + 1) = 0. ! vorticity [1/s]
    !
    real :: pi                      ! pi
    real :: beta                    ! beta parameter
    real :: dx                      ! gridpoint distance in x [m]
    real :: dy                      ! gridpopint distance in y [m]
    real :: rossby                  ! Rossby number
    real :: f0                      ! corriolis par. at cent.latitude [1./s]
    real :: delt2                   ! 2*delt [s]
    !
end module baromod
!
!----------------------------------------------------------------------
!
program baro
    use baromod
    !
    !
    !     program baro organizes the model run by calling individual subroutines
    !     for initialization, time stepping, output and finalization
    !
    integer :: jstep     ! loop index
    !
    !     a) initialization
    !
    call baroini
    !
    !     b) time stepping + output
    !
    do jstep = 1, nrun
        call barostep
        nstep = nstep + 1
        if(mod(nstep, nout)==0) call baroout
    enddo
    !
    !     c) finalizing
    !
    call barostop
    !
    stop
end
!
!----------------------------------------------------------------------
!
subroutine baroini
    use baromod
    !
    !     subroutine baroini initializes the run
    !
    !
    !
    real :: zlat        ! central latitude in rad
    real :: zxch        ! channel length in m
    real :: zych        ! channel width in m
    real :: zdelt       ! short initial time step for euler
    integer :: kits = 3 ! no short initial time steps
    !
    !     set some constants
    !
    pi = 2. * asin(1.)
    zlat = pi * rlat / 180.
    zxch = pi * radea * cos(zlat) * xchannel / 180.
    zych = pi * radea * ychannel / 180.
    !
    !     define grid (by dx,dy)
    !
    dx = zxch / real(nx)
    dy = zych / real(ny + 1)
    !
    !     set coriolis parameter amd beta according to central latitude
    !
    f0 = 2. * omega * sin(zlat)
    beta = 2. * omega * cos(zlat) / radea
!    beta = 0.
    !
    !     print grid information
    !
    print*, ' f0       = ', f0
    print*, ' beta     = ', beta
    print*, ' delta x  = ', dx
    print*, ' delta y  = ', dy
    !
    !     compute rossby wave phase velocity and related courant number
    !
    print*, ' Rossby Cp   = ', beta * zxch * zxch / (2. * pi)**2
    print*, ' Rossby C-No = ', (beta * zxch * zxch / (2. * pi)**2) * delt / dx
    !
    !     set delt2 for leapfrog
    !
    delt2 = 2. * delt
    !
    !     open output file
    !
    open(10, file = 'barogp', form = 'unformatted')
    !
    !     set the initial field(s)
    !
    call initial
    !
    !     do initial (short) euler time steps to obtain values at t=delt
    !
    zdelt = delt / 2**(kits)
    do js = 1, kits
        zdelt = zdelt + zdelt
        call mkvort(s, vo, dx, dy, NX, NY)
        call tendency
        call euler(sm, dsdt, s, zdelt, NX, NY)
        call boundary(s, NX, NY)
    enddo
    !
    return
end
!
!----------------------------------------------------------------------
!
subroutine barostep
    use baromod
    real :: zu(NX, NY), zv(NX, NY) ! u and v (velocities)
    !
    !     subroutine barostep does the time steping
    !
    !     a) compute vorticity from streamfunction
    !
    call mkvort(s, vo, dx, dy, NX, NY)
    !
    !     b) compute streamfunction tendencies
    !
    call tendency
    !
    !     c) do the leapfrog timestep
    !           Therefore first check with Courant Friedrich Levi if it works!
    !
    !     comput u and v from streamfunction
    !
    call mkuv(s, zu, zv, dx, dy, NX, NY)
    !
    call CFL(zu, zv, dx, dy, delt, NX, NY)  ! Courant Friedrich Levi
    call leapfrog(s, sm, dsdt, delt2, NX, NY)
    !
    !     d) apply boundary conditions to streamfunction
    !
    call boundary(s, NX, NY)
    !
    return
end
!
!----------------------------------------------------------------------
!
subroutine barostop
    use baromod
    !
    !     subroutine barostop finelizes the run
    !
    !     close output files
    !
    close(10)
    !
    !     write grads control file
    !
    call gradsinfo
    !
    print*, '****************************************************'
    print*, 'Run finished at nstep= ', nstep
    print*, '****************************************************'
    !
    return
end
!
!----------------------------------------------------------------------
!
subroutine baroout
    use baromod
    !
    !     subroutine baroout writes fields to output file
    !
    real :: zu(NX, NY), zv(NX, NY) ! u and v (velocities)
    !
    !     comput u and v from streamfunction
    !
    call mkuv(s, zu, zv, dx, dy, NX, NY)
    !
    !     write u
    !
    write(10) zu(1 : NX, 1 : NY)
    !
    !     write v
    !
    write(10) zv(1 : NX, 1 : NY)
    !
    !     write streamfct.
    !
    write(10) s(1 : NX, 1 : NY)
    !
    !     write geopotential height
    !
    write(10) s(1 : NX, 1 : NY) * f0 / 9.81
    !
    !     write vorticity
    !
    write(10) vo(1 : NX, 1 : NY)
    !
    !     advance counter
    !
    nwout = nwout + 1
    !
    return
end
!
subroutine mkuv(ps, pu, pv, pdx, pdy, kx, ky)
    implicit none
    !
    !     subroutine mkuv computs u and v from streamfunction
    !
    integer :: kx                  ! x-dimension
    integer :: ky                  ! y-dimension
    real :: pdx                    ! grid point distance x [m]
    real :: pdy                    ! grid point distance y [m]
    real :: ps(0 : kx + 1, 0 : ky + 1)      ! streamfunction [m**2/s]
    real :: pu(kx, ky)              ! u [m/s]
    real :: pv(kx, ky)              ! v [m/s]
    integer :: i, j                 ! loop indizes
    !
    do j = 1, ky
        do i = 1, kx
            pu(i, j) = -0.5 * (ps(i, j + 1) - ps(i, j - 1)) / pdy
            pv(i, j) = 0.5 * (ps(i + 1, j) - ps(i - 1, j)) / pdx
        enddo
    enddo
    !
    return
end
!
!----------------------------------------------------------------------
!
subroutine tendency
    use baromod
    !
    !     subroutine tendency computes the streamfunction tendency
    !
    real :: zdt(0 : NX + 1, 0 : NY + 1) ! vorticity tendency
    real :: zj(0 : NX + 1, 0 : NY + 1)  ! jacobian (advection of rel vort.)
    real :: zv(0 : NX + 1, 0 : NY + 1)  ! y-velocity v (ds/dx)
    !
    !     compute jacobian J(s,vo) (adv. of rel. vort.)
    !
    call jacobi(s, vo, zj, dx, dy, NX, NY)
    !
    !     compute v (ds/dx)
    !
    call mkdfdx(s, zv, dx, NX, NY)
    !
    !     add advection of rel. and planetray vorticity to get vort. tendency
    !
    zdt(:, :) = -zj(:, :) - beta * zv(:, :)
    !
    !     compute inverse Laplacian to get streamfunction tendency
    !
    call sor(zdt, dsdt, dx, dy, NX, NY)
    !
    return
end
!
!----------------------------------------------------------------------
!
subroutine mkvort(ps, pvo, pdx, pdy, kx, ky)
    implicit none
    !
    !     subroutine mkvort computes the vorticity from a given streamfunction
    !
    integer :: kx                 ! x-dimension
    integer :: ky                 ! y-dimension
    real :: pdx                   ! x grid distance
    real :: pdy                   ! y grid distance
    real :: ps(0 : kx + 1, 0 : ky + 1)     ! input: stream function
    real :: pvo(0 : kx + 1, 0 : ky + 1)    ! output: vorticity
    integer :: i, j                ! loop indizes
    !
    !     compute Laplacian
    !
    call laplace(ps, pvo, pdx, pdy, kx, ky)
    !
    !     make boundary conditions
    !
    call boundary(pvo, kx, ky)
    !
    return
end
!
!-----------------------------------------------------------------------     
!
subroutine boundary(ps, kx, ky)
    implicit none
    !
    !     subroutine boundary set boundary conditions (cyclic in x)
    !
    integer :: kx, ky   ! dimensions
    real :: ps(0 : kx + 1, 0 : ky + 1) ! input field
    !
    ps(0, :) = ps(kx, :)
    ps(kx + 1, :) = ps(1, :)
    ps(:, 0) = 0.
    ps(:, ky + 1) = 0.
    !
    return
end
!
!----------------------------------------------------------------------     
!
subroutine initial
    use baromod
    !
    !     subroutine initial sets the initial streamfunktion field
    !     here: a wave with wave numbers zk and zl (see below)
    !
    real :: za = 10. * 1.E6 ! initial amplitude [m**2/s]
    real :: zk = 1.       ! wave number in x-direction
    real :: zl = 0.5      ! wave number in y-direction
    integer :: i, j        ! loop indizes
    !
    !     set initial streamfunction
    !
    do j = 1, NY
        do i = 1, NX
            s(i, j) = za * sin(2. * pi * real(i - 1) * zk / real(NX))                      &
                    & * sin(2. * pi * real(j) * zl / real(NY + 1))
        enddo
    enddo
    !
    !     set boundary conditions
    !
    call boundary(s, NX, NY)
    !
    !     copy to sm
    !
    sm(:, :) = s(:, :)
    !
    !     compute vorticity and write initial field to output
    !
    call mkvort(s, vo, dx, dy, NX, NY)
    call baroout
    !
    return
end
!
!-----------------------------------------------------------------------     
!
subroutine gradsinfo
    use baromod
    !
    real :: zy1, zdy
    !
    !     open file
    !
    open(10, file = 'barogp.ctl', form = 'formatted')
    !
    !     write grads control file
    !
    zdy = ychannel / real(ny + 1)
    zy1 = rlat - ychannel / 2. + zdy
    write(10, 1) 'DSET ^barogp'
    write(10, 1) 'UNDEF 9E+09'
    write(10, 1) 'OPTIONS SEQUENTIAL'
    write(10, 2) 'XDEF ', NX, ' LINEAR ', 0., ' ', xchannel / real(NX)
    write(10, 2) 'YDEF ', NY, ' LINEAR ', zy1, ' ', zdy
    write(10, 1) 'ZDEF 1 LINEAR 1 1'
    write(10, 3) 'TDEF ', nwout, ' LINEAR 00:00Z01jan0001 20mn'
    write(10, 1) 'VARS 5'
    write(10, 1) 'U 0 99 zonal velocity'
    write(10, 1) 'V 0 99 meridional velocity'
    write(10, 1) 'S 0 99 streamfunction'
    write(10, 1) 'Z 0 99 geopotential height'
    write(10, 1) 'VO 0 99 vorticity'
    write(10, 1) 'ENDVARS'
    !
    !     close file
    !
    close(10)
    !
    return
    1 format(A)
    2 format(A, I5, A, F10.5, A, F10.5)
    3 format(A, I5, A)
end
!
!=======================================================================
!  
!     the folowing subroutines are just dummies and need to be
!     replaced 
!
      subroutine sor(pdf,pf,pdx,pdy,kx,ky)
      implicit none
!
!     subroutine sor computes the inverse Laplacian from a given field
!     by using the Successive OverRelaxation method (SOR)
!
      integer, intent(in) :: kx                   ! x dimension
      integer, intent(in) :: ky                   ! y dimension
      
      integer :: iter                             ! number of iterations needed
      integer :: i, j, k                          ! loop indices
      
      real, intent(in)    :: pdx                  ! x grid point distance
      real, intent(in)    :: pdy                  ! y grid point distance
      real, intent(in)    :: pdf(0:kx+1,0:ky+1)   ! input: field
      real, intent(inout) :: pf(0:kx+1,0:ky+1)    ! output: inverse Laplacian of input

      real :: zerr = 0.                           ! error at every grid point
      real :: zacc_ini = 0.                       ! initial accuracy = max zerr initial
      real :: zacc = 0.                           ! accuracy = max zerr
      real, parameter :: omega = 1.5              ! over correction
      real, parameter :: eps = 1.E-4              ! minimal error reduction
!
      ! make sure of the right boundaries
      call boundary(pf,kx,ky)
      call boundary(pdf,kx,ky)

      ! calculate the number of iterations
      iter = kx * ky * abs(log10(eps)) / 3

      ! calculate the initial error
      do j=ky,1,-1
        do i=1,kx
          zerr =  (1 / (pdx*pdx) * (pf(i+1,j) + pf(i-1,j) - 2 * pf(i,j))        &
          &     +  1 / (pdy*pdy) * (pf(i,j+1) + pf(i,j-1) - 2 * pf(i,j))       &
          &     - pdf(i,j))

          zacc_ini = zacc_ini + (abs(zerr) / (kx*ky))
        end do
      end do

      ! calculate the sor
      do k = 1,iter
        zacc = 0.
        do j=ky,1,-1
          do i=1,kx
            zerr = (1 / (pdx*pdx) * (pf(i+1,j) + pf(i-1,j) - 2 * pf(i,j))      &
            &     +  1 / (pdy*pdy) * (pf(i,j+1) + pf(i,j-1) - 2 * pf(i,j))     &
            &     - pdf(i,j))
            
            pf(i,j) = pf(i,j) + omega * zerr / (2 / (pdx*pdx) + 2 / (pdy*pdy))

            zacc = zacc + (abs(zerr) / (kx*ky))
          end do
        end do

        if (zacc <= zacc_ini*eps) then
          exit
        end if
      end do

      ! debug manually...
      !print*, k*1.0/iter*1.0,' percent of maximal iterations needed'

      call boundary(pdf,kx,ky)

      return
      end
!
!-----------------------------------------------------------------------
!
subroutine mkdfdx(pf, pdfdx, pdx, kx, ky)
    implicit none
    !
    !     subroutine mkdfdx computes x derivation from a field
    !     using central differences
    !
    integer :: kx                 ! x-dimension
    integer :: ky                 ! y-dimension
    real :: pdx                   ! x grid distance
    real, intent(in) :: pf(0 : kx + 1, 0 : ky + 1)     ! input: field
    real, intent(out) :: pdfdx(0 : kx + 1, 0 : ky + 1)  ! output: dfield/dx
    !
    integer :: i, j ! loop variable

    do j = 1, ky
        do i = 1, kx
            pdfdx(i, j) = (pf(i + 1, j) - pf(i - 1, j)) / (2. * pdx)
        end do
    end do
    return
end
!
!-----------------------------------------------------------------------
!
subroutine jacobi(p1, p2, pj, pdx, pdy, kx, ky)
    integer :: kx              ! x-dimension
    integer :: ky              ! y-dimension
    real :: p1(0 : kx + 1, 0 : ky + 1)  ! input: field 1
    real :: p2(0 : kx + 1, 0 : ky + 1)  ! input: field 2
    real :: pj(0 : kx + 1, 0 : ky + 1)  ! output: jacobi(1,2)
    real :: pdx                ! x grid distance
    real :: pdy                ! y grid distance
    integer :: i
    integer :: j
    real :: J1(0 : kx + 1, 0 : ky + 1)
    real :: J2(0 : kx + 1, 0 : ky + 1)
    real :: J3(0 : kx + 1, 0 : ky + 1)
    real :: J4(0 : kx + 1, 0 : ky + 1)
    real, parameter :: f = 1. / 3.

    do j = 1, ky
        do i = 1, kx
            J1(i, j) = (1 / (4 * pdx * pdy)) * (((p1(i + 1, j) - p1(i - 1, j)) * (p2(i, j + 1) - &
                    & p2(i, j - 1))) - ((p1(i, j + 1) - p1(i, j - 1)) * (p2(i + 1, j) - p2(i - 1, j))))

            J2(i, j) = (1 / (4 * pdx * pdy)) * ((p1(i + 1, j) * (p2(i + 1, j + 1) - p2(i + 1, j - 1))) - &
                    & (p1(i - 1, j) * (p2(i - 1, j + 1) - p2(i - 1, j - 1))) - (p1(i, j + 1) * (p2(i + 1, j + 1) - &
                    & p2(i - 1, j + 1))) + (p1(i, j - 1) * (p2(i + 1, j - 1) - p2(i - 1, j - 1))))

            J3(i, j) = (1 / (4 * pdx * pdy)) * ((p2(i, j + 1) * (p1(i + 1, j + 1) - p1(i - 1, j + 1))) - &
                    & (p2(i, j - 1) * (p1(i + 1, j + 1) - p1(i - 1, j - 1))) - (p2(i + 1, j) * (p1(i + 1, j + 1) - &
                    & p1(i + 1, j - 1))) + (p2(i - 1, j) * (p1(i - 1, j + 1) - p1(i - 1, j - 1))))

            pj(i, j) = f * (J1(i, j) + J2(i, j) + J3(i, j))
        enddo
    enddo

    return
end

!
!-----------------------------------------------------------------------
!
subroutine laplace(pf, pdf, pdx, pdy, kx, ky)
    implicit none
    !
    !     subroutine laplace computes the laplacian from a field
    !
    integer :: kx                 ! x-dimension
    integer :: ky                 ! y-dimension
    real :: pdx                   ! x grid distance
    real :: pdy                   ! y grid distance
    real :: pf(0 : kx + 1, 0 : ky + 1)     ! input: field
    real :: pdf(0 : kx + 1, 0 : ky + 1)    ! output: Laplacian entspricht G ?!
    integer :: i, j                ! counter for loops
    !
    call boundary(pf, kx, ky)      ! set boundary values

    do j = 1, ky
        do i = 1, kx
            pdf(i, j) = 1 / (pdx * pdx) * (pf(i + 1, j) + pf(i - 1, j) - 2 * pf(i, j)) &
                    & + 1 / (pdy * pdy) * (pf(i, j + 1) + pf(i, j - 1) - 2 * pf(i, j))
        enddo
    enddo

    call boundary(pdf, kx, ky)      ! set new boundary values

    return
end
!

!
!-----------------------------------------------------------------------
!
subroutine euler(pfm, pdfdt, pf, pdelt, kx, ky)
    implicit none
    !
    !     subroutine euler does an explicit Euler time step
    !
    integer, intent(in) :: kx                   ! x dimension
    integer, intent(in) :: ky                   ! y dimension
    real, intent(in) :: pdelt                ! time step [s]
    real, intent(in) :: pfm(0 : kx + 1, 0 : ky + 1)   ! input f(t)
    real, intent(in) :: pdfdt(0 : kx + 1, 0 : ky + 1) ! input tendency
    real, intent(out) :: pf(0 : kx + 1, 0 : ky + 1)    ! output f(t+1)
    !
    integer :: i, j ! loop variables

    do j = 1, ky
        do i = 1, kx
            pf(i, j) = pfm(i, j) + pdelt * pdfdt(i, j)
        enddo
    enddo

    return

end
!
!-----------------------------------------------------------------------
!
subroutine leapfrog(pf, pfm, pdfdt, pdelt2, kx, ky)
    implicit none
    !
    !     subroutine leapfrog does a leapfrog time step
    !     with Robert Asselin filter
    !
    integer :: kx                ! x dimension
    integer :: ky                ! y dimension
    real :: pdelt2               ! 2.* timestep
    real :: pf(0 : kx + 1, 0 : ky + 1)    ! input/output f(t)
    real :: pf_star(0 : kx + 1, 0 : ky + 1)    ! pf*
    real, intent(inout) :: pfm(0 : kx + 1, 0 : ky + 1)   ! input/output f(t-1) (filtered)
    real :: pdfdt(0 : kx + 1, 0 : ky + 1) ! input tendency
    real :: pf_temp(0 : kx + 1, 0 : ky + 1) !temporary t+1

    integer :: i, j  !loop variable

    real :: gamma = 0.1  ! Rober-Asselin filter
    !
    do j = 1, ky
        do i = 1, kx
            pf_temp(i, j) = pfm(i, j) + pdelt2 * pdfdt(i, j)
            pf_star(i, j) = pf(i, j) + gamma * (pfm(i, j) - 2 * pf(i, j) + pf_temp(i, j))
        end do
    end do

    pfm = pf_star
    pf = pf_temp
    return
end subroutine leapfrog

subroutine CFL(pu, pv, pdx, pdy, pdt, kx, ky)
    implicit none

    real :: pdx, pdy ! gridwidth
    real :: pdt ! timestep
    integer :: kx, ky ! grid

    real :: pu(kx, ky), pv(kx, ky)   ! speed

    integer :: i, j ! loop variables

    do j = 1, ky
        do i = 1, kx
            if (pu(i, j) * pdt / pdx > 1) then
                write(*, *) "Courant Levi Criterion not fullfilled"
                write(*,*) " pu = ", pu(i,j), "|pv = ", pv(i,j)
                write(*,*) "At i = " , i, "| j = ", j
                stop
            else if (pv(i, j) * pdt / pdy > 1) then
                write(*, *) "Courant Levi Criterion not fullfilled"
                write(*,*) " pu = ", pu(i,j), "|pv = ", pv(i,j)
                write(*,*) "At i = " , i, "| j = ", j
                stop
            endif
        end do
    end do
    return
end subroutine CFL
