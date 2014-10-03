       MODULE BULK
!     *********************************************************************************
       IMPLICIT NONE

!     ---------------------------------------------------------------------------------
!     L -индекс геометрии: L=2 - цилиндрические координаты, L=1 - декартовы координаты
!     N1 - число узлов расчетной сетки в направлении X1
!     N2 - число узлов расчетной сетки в направлении X2
!     N3 - число узлов расчетной сетки в направлении X3
!     NSTOP - Полное число шагов по времени
!     NPRINT - Интервал печати
!     NSTEP - Текущий шаг по времени

      INTEGER L,N1,N2,N3,NSTOP, NPRINT,NSTEP,N1G,N2G,N3G
!     ---------------------------------------------------------------------------------      
!     X1(:) - координаты узлов по оси X1       
!     X2(:) - координаты узлов по оси X2         
!     X3(:) - координаты узлов по оси X3 

       REAL(8), POINTER :: X1(:),X2(:),X3(:)              
!     ----------------------------------------------------------------------------------
!     ROCON(:,:,:) - консерватиные переменные плотности на текущем временном слое
!     U1CON(:,:,:) - консервативная скорость по оси X1 на текущем временном слое
!     U2CON(:,:,:) - консервативная скорость по оси X2 на текущем временном слое
!     U3CON(:,:,:) - консервативная скорость по оси X3 на текущем временном слое
!     TCON(:,:,:) - консервативная переменная температуры на текущем временном слое
       
       REAL(8), POINTER :: ROCON(:,:,:),U1CON(:,:,:),U2CON(:,:,:), &
      U3CON(:,:,:), TCON(:,:,:)
!     -----------------------------------------------------------------------------------
!     RONCON(:,:,:) - консерватиные переменные плотности на следующем временном слое
!     U1NCON(:,:,:) - консервативная скорость по оси X1 на следующем временном слое
!     U2NCON(:,:,:) - консервативная скорость по оси X2 на следующемвременном слое
!     U3NCON(:,:,:) - консервативная скорость по оси X3 на следующем временном слое
!     TNCON(:,:,:) - консервативная переменная температуры на следующем временном слое

       REAL(8), POINTER :: RONCON(:,:,:),U1NCON(:,:,:),U2NCON(:,:,:), &
      U3NCON(:,:,:),TNCON(:,:,:)
!     ----------------------------------------------------------------------------------
!     P1(:,:,:) -  давление на гранях, перпендикулярных оси X1
!     RO1(:,:,:) - плотность на гранях, перпендикулярных оси X1 
!     U11(:,:,:) - скорость по оси X1 на гранях, перпендикулярных оси X1 
!     U21(:,:,:) - скорость по оси X2 на гранях, перпендикулярных оси X1 
!     U31(:,:,:) - скорость по оси X3 на гранях, перпендикулярных оси X1 
!     T1(:,:,:) -  температура на гранях, перпендикулярных оси X1
    
        REAL(8), POINTER :: P1(:,:,:),RO1(:,:,:),U11(:,:,:), &
      U21(:,:,:),U31(:,:,:),T1(:,:,:)
!     ------------------------------------------------------------------------------------
!     P2(:,:,:) -  давление на гранях, перпендикулярных оси X2
!     RO2(:,:,:) - плотность на гранях, перпендикулярных оси X2 
!     U12(:,:,:) - скорость по оси X1 на гранях, перпендикулярных оси X2 
!     U22(:,:,:) - скорость по оси X2 на гранях, перпендикулярных оси X2 
!     U32(:,:,:) - скорость по оси X3 на гранях, перпендикулярных оси X2
!     T2(:,:,:) -  температура на гранях, перпендикулярных оси X2

         REAL(8), POINTER :: P2(:,:,:),RO2(:,:,:),U12(:,:,:), &
      U22(:,:,:),U32(:,:,:),T2(:,:,:)      
!     ------------------------------------------------------------------------------------
!     P3(:,:,:) -  давление на гранях, перпендикулярных оси X3
!     RO3(:,:,:) - плотность на гранях, перпендикулярных оси X3 
!     U13(:,:,:) - скорость по оси X1 на гранях, перпендикулярных оси X3 
!     U23(:,:,:) - скорость по оси X2 на гранях, перпендикулярных оси X3 
!     U33(:,:,:) - скорость по оси X3 на гранях, перпендикулярных оси X3
!     T3(:,:,:) -  температура на гранях, перпендикулярных оси X3

         REAL(8), POINTER :: P3(:,:,:),RO3(:,:,:),U13(:,:,:), &
      U23(:,:,:),U33(:,:,:),T3(:,:,:)   
!     -----------------------------------------------------------------------------------
!     R(:)- одномерный буферный массив для хранения вычисляемых значений инварианта Римана R
!     Q(:)- одномерный буферный массив для хранения вычисляемых значений инварианта Римана Q

      REAL(8), POINTER :: RBUF(:),QBUF(:),TFBUF(:),TBBUF(:), &
      U2FBUF(:),U2BBUF(:),U3FBUF(:),U3BBUF(:)
     
!     -----------------------------------------------------------------------------------
!     Q1(:,:,:) -  массив потоков по аправлению X1
!     Q2(:,:,:) -  массив потоков по аправлению X2
!     Q3(:,:,:) -  массив потоков по аправлению X3
      REAL(8), POINTER :: Q1(:,:,:), Q2(:,:,:), Q3(:,:,:)

!     ************************************************************************************* 
       END MODULE BULK
       
       MODULE FORCES
!     *********************************************************************************
       IMPLICIT NONE
 !    ---------------------------------------------------------------------------------------
 !    F1(:,:,:) - силы (трения и др) в направлении оси X1
 !    F2(:,:,:) - силы (трения и др) в направлении оси X2
 !    F3(:,:,:) - силы (трения и др) в направлении оси X3
       
       REAL(8), POINTER :: F1(:,:,:),F2(:,:,:),F3(:,:,:)
       
       END MODULE FORCES

      MODULE STRESS
!     *********************************************************************************
       IMPLICIT NONE
!     --------------------------------------------------------------------------------------
!     SIGMA11(:,:,:) - напряжение трения в направлении X1 для грани, перпендикулярной оси X1
!     SIGMA21(:,:,:) - напряжение трения в направлении X2 для грани, перпендикулярной оси X1
!     SIGMA31(:,:,:) - напряжение трения в направлении X3 для грани, перпендикулярной оси X1  
!     SIGMA12(:,:,:) - напряжение трения в направлении X1 для грани, перпендикулярной оси X2
!     SIGMA22(:,:,:) - напряжение трения в направлении X2 для грани, перпендикулярной оси X2
!     SIGMA32(:,:,:) - напряжение трения в направлении X3 для грани, перпендикулярной оси X2
!     SIGMA13(:,:,:) - напряжение трения в направлении X1 для грани, перпендикулярной оси X3
!     SIGMA23(:,:,:) - напряжение трения в направлении X2 для грани, перпендикулярной оси X3
!     SIGMA33(:,:,:) - напряжение трения в направлении X3 для грани, перпендикулярной оси X3
     
       REAL(8), POINTER :: SIGM11(:,:,:),SIGM21(:,:,:),SIGM31(:,:,:), &
       SIGM12(:,:,:),SIGM22(:,:,:),SIGM32(:,:,:),SIGM13(:,:,:), &
       SIGM23(:,:,:),SIGM33(:,:,:)
       
       END MODULE STRESS

      MODULE CONSTANT
      IMPLICIT NONE
!     -----------------------------------------------------------------------------------       
!      SOUND - скорость звука
!      RO0G - невозмущенная плотность жидкости
!      RO0S - невозмущенная плотность вещества преграды
!      DT - шаг по времени
!      VIS - кинематическая вязкость
!      U10 - Начальная скорость по оси X1
!      U20 - Начальная скорость по оси X2
!      U30 - Начальная скорость по оси X3
!      RO0G - Невозмущенная плотность газа
!      RO0S - Невозмущенная плотность твердого тела 
!      T0 - Начальная температура 
!      TIME - Текущее время
!      CFL- число Куранта

      REAL(8) SOUND, RO0G, RO0S, DT, VIS, U10, U20, U30,TIME,CFL, &
      POUTLET, U3INLET,U2INLET,U1INLET,TINLET 
      REAL(8) Kappa,Cv,FQ,ALFA,TTop,TBottom, &
      XLW,XLE,XLB,XLT,XLS,X1E,X1W,X2N,X2S,X3B,X3T, &
      TCN,TE,TN,TS,TT,TW,TB,TC,T0,TBN,TF,TFN,TMAXB, &
      TMAXF,TMINB,TMINF, &
      U1C,U1B,U1E,U1N,U1FN,U1S,U1T,U1W,U1BN,U1CN,U1CP, &
      U1F,U1MAXB,U1MAXF,U1MINB,U1MINF, &
      U2S,U2T,U2W,U2BN,U2CN,U2F,U2FN,U2MAXB,U2MAXF, &
      U2MINB,U2MINF,U2B,U2C,U2E,U2N,U2CP, &
      U3B,U3C,U3E,U3N,U3S,U3T,U3W,U3BN,U3CN,U3F,U3FN, &
      U3MAXB,U3MAXF,U3MINB,U3MINF, &
      UB,UC,UCB,UCF,UCN,UF,UN, &
      ROCN,ROC,ROW,ROCL,ROE,RON,ROS,ROT,ROB,RO0,RO0F, &
      ROC0,ROF,RO0B, &
      RB,RF,RFN,RMAX,RMIN,RN,RC,RCN, &
      DVC,DS1,DS2,DS3,DX1,DX2,DX3,DX, &
      PW,PN,PS,PT,PB,PE,PF,PC,PCN, &
      QF,QMAX,QMIN,QN,QB,QBN,QC,QCN, &
      GU2F,GU3B,GU3F,GQ,GR,GTB,GTF,GU1B,GU1F,GU2B, &
      DTV,DTV1,DTV2,DTV3,DTU,DTU1,DTU2,DTU3, &
      X1C,X1L,SNDRCVTIME, SNDRCVTIME_T,Tmp, &
      GT,GU1,GU2,GU3,TMAX,TMIN,U1MAX,U1MIN,U2MAX,U2MIN,U3MAX,U3MIN, &
      DELTA
	  
      REAL(8), PARAMETER :: PI = 3.141592653589793239

      INTEGER myid, numprocs, status, ierror,px,py,pz, &
      left, right, up, down, near,far, new_comm, coords(0:2)
      END MODULE CONSTANT

	  
      PROGRAM MOZER
      USE FORCES
      USE BULK
      USE CONSTANT	  
      implicit none
      include 'mpif.h'
!      integer PX,PY,PZ,ierror,numprocs,ndims,dim_size(0:2),myid,right, &
!                left,down,up,near,far,new_comm,coords(0:2),c1,c2,c3,ii
      integer c1,c2,c3,ii,dim_size(0:2),ndims,NCOUNT
      logical periods(0:2),reorder
	  real*8 dtmin
      call MPI_INIT(ierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierror)
      PX = 1
      PY = 1
      PZ = 1
      do while (PX*PY*PZ < numprocs)
            PZ = 2*PZ
            if (PX*PY*PZ < numprocs) then
                PY = 2*PY
                if  (PX*PY*PZ < numprocs) then
                    PX=2*PX
                endif
            endif
      enddo

      ndims = 3                   ! 3D Px x Py x Pz
      dim_size(0) = PX             ! X dim size
      dim_size(1) = PY             ! Y dim size
      dim_size(2) = PZ             ! Z dim size
      periods(0) = .false.         ! periodic on X axis
      periods(1) = .true.         ! periodic on Y axis
      periods(2) = .false.         ! periodic on Y axis
      reorder = .true.            ! reorder is true

      call MPI_Cart_create(MPI_COMM_WORLD, ndims, dim_size, periods, reorder, new_comm, ierror)
      call MPI_COMM_RANK(new_comm, myid, ierror)
      call MPI_Cart_Coords(new_comm, myid, ndims, coords, ierror)

      call MPI_Cart_Shift(new_comm, 0, 1, left, right, ierror)
      call MPI_Cart_Shift(new_comm, 1, 1, down, up, ierror)
      call MPI_Cart_Shift(new_comm, 2, 1, near, far, ierror)
      
!      write(*,'(a,7(i4,2x))') 'myid,r,l,u,d,n,f=',myid,right,left,down,up,near,far
!      write(*,'(a,4(i4,2x))') 'myid,coords(0:2)=',myid,coords(0:2)
!      if(myid.eq.0)then
!          do ii=1,numprocs-1
!            c1=ii/(PY*PZ)
!            c2=(ii-c1*PY*PZ)/PZ
!            c3=ii-c1*PY*PZ-c2*PZ
!            print *, 'ii,c1,c2,c3=',ii,c1,c2,c3
!          enddo
!      endif
      call INPUT
!	  write(*,'(a,3(i3,2x))') 'myid,N1,N2,N3=',myid,N1,N2,N3
      call INITIAL_DATE
!	  call write_data_mpi
!	******************************************************************************************
!	  dt=1.d10
!	  call mpi_barrier(mpi_comm_world,ierror)
!      CALL TIMESTEPSIZE
!	  call mpi_allreduce(dtmin,dt,1,mpi_double_precision,mpi_min,new_comm,ierror)
!	  dt=dtmin
      if(myid.eq.0)read(*,*)
      call write_data_mpi
      NCOUNT=0
      TIME=0.d0
      dt=1.d-4
100   CONTINUE
      CALL TIME_STEP_SIZE
      NSTEP=NSTEP+1
      NCOUNT=NCOUNT+1
      TIME=TIME+0.5*DT
!	  write(*,*) 'myid, dt=',myid,dt!
!	  call mpi_barrier(new_comm,ierror)
!	  goto 200
      CALL PHASE1
!      CALL NEWT
      CALL STRESSTENSOR
      CALL USEFORCES
      CALL PHASE2
      CALL PHASE1
!      CALL NEWT
      CALL USEFORCES
      TIME=TIME+0.5*DT
      IF(NCOUNT.LT.NPRINT) THEN
		GOTO 100
      ELSE
		NCOUNT=0
        if(myid.eq.0)PRINT *,NSTEP,DT 
        call write_data_mpi
        IF(NSTEP.EQ.NSTOP) GO TO 200
      ENDIF
      GOTO 100
     
200   continue
!      call write_data_mpi
!	******************************************************************************************
	  if(myid.eq.0)write(*,*) 'Program has been done successfully'
      call MPI_FINALIZE(ierror)
	  contains
      SUBROUTINE  INPUT
      USE BULK
      USE FORCES
      USE CONSTANT
      USE STRESS
	  implicit none
	  include 'mpif.h'
	  integer i,j,k
      INTEGER:: NMAX
!     *************************************************************************************
!     Блок ЗАДАНИЯ РАСЧЕТНОЙ ОБЛАСТИ, ТИПА ГЕОМЕТРИИ И РАСЧЕНОЙ СЕТКИ
      L=2       ! Индекс геометрии
      
      N1G=17  !9     ! Число расчетных узлов по направлению X1
      N2G=33 !3     ! Число расчетных узлов по направлению X2
      N3G=65  !27     ! Число расчетных узлов по направлению X3
      N1=(N1G-1)/PX+1
      N2=(N2G-1)/PY+1
      N3=(N3G-1)/PZ+1
      if(coords(0).eq.PX-1)N1=N1G-(N1-1)*coords(0)
      if(coords(1).eq.PY-1)N2=N2G-(N2-1)*coords(1)
      if(coords(2).eq.PZ-1)N3=N3G-(N3-1)*coords(2)
      DELTA=1.
      X1W=0.15D0    ! Положение Восточной границы области
      X1E=0.25D0  !2.D0*PI  !2.*DELTA    ! Положение Западной границы области

      X2S=0         ! Положение Южной границы области
      X2N=2.D0*PI    !*DELTA     ! Положение Северной границы области

      X3B=0.    ! Положение Нижней границы области
      X3T=1.D0  !2.D0*PI   !*DELTA     ! Положение Верхней границы области

      TTop = 80     ! Температура на верхней грани
      TBottom = 10  ! Температура на нижней грани
      
      Kappa = 1.    ! Коэф. Каппа

      NSTEP  =0   ! Счетчик шагов по времени
      NPRINT =100  ! Интервал печати
      NSTOP  =100
      TIME   =0.
      CFL    =0.2
!     -------------------------------------------------------------------------------------------------
      ALLOCATE (X1(0:N1+1),X2(0:N2+1),X3(0:N3+1))


      X1(1)=X1W+(X1E-X1W)/PX*coords(0)
      X2(1)=X2S+(X2N-X2S)/PY*coords(1)
      X3(1)=X3B+(X3T-X3B)/PZ*coords(2)

      DX1=(X1E-X1W)/PX/(N1-1)
      DX2=(X2N-X2S)/PY/(N2-1)
      DX3=(X3T-X3B)/PZ/(N3-1)

      DO I=1,N1
      X1(I+1)=X1(I)+DX1
      ENDDO
      X1(0)=X1(1)-DX1

      DO J=1,N2
      X2(J+1)=X2(J)+DX2
      ENDDO
      X2(0)=X2(1)-DX2

      DO K=1,N3
      X3(K+1)=X3(K)+DX3
      ENDDO
      X3(0)=X3(1)-DX3

!     КОНЕЦ блока ЗАДАНИЯ РАСЧЕТНОЙ ОБЛАСТИ, ТИПА ГЕОМЕТРИИ И РАСЧЕНОЙ СЕТКИ

!     REm=5600
      VIS=0.5/50.  ! 0.5  !Кинематический коэффициент вязкости
      T0=1.   ! Начальная температура

      U10=0.     ! Начальная скорость по оси X1
      U20=0.     ! Начальная скорость по оси X2
      U30=1.    ! Начальная скорость по оси X3
      SOUND=10. ! Скорость звука

      POUTLET=0          !  ДАВЛЕНИЕ НА ВЕРХНЕЙ ГРАНИЦЕ
      U3INLET=U30   !U30        !  СКОРОСТЬ НА НИЖНЕЙ ГРАНИЦЕ
      U2INLET=U20        !  СКОРОСТЬ НА НИЖНЕЙ ГРАНИЦЕ
      U1INLET=U10        !  СКОРОСТЬ НА НИЖНЕЙ ГРАНИЦЕ  
      TINLET=T0          !  Температура НА НИЖНЕЙ ГРАНИЦЕ
      RO0G=1.   ! Невозмущенная плотность газа
      RO0S=100000000. ! Невозмущенная плотность твердого тела

      ALLOCATE (ROCON(0:N1,0:N2,0:N3),TCON(0:N1,0:N2,0:N3), &
      U1CON(0:N1,0:N2,0:N3),U2CON(0:N1,0:N2,0:N3),U3CON(0:N1,0:N2,0:N3))

      ALLOCATE (RONCON(0:N1,0:N2,0:N3),TNCON(0:N1,0:N2,0:N3), &
      U1NCON(0:N1,0:N2,0:N3),U2NCON(0:N1,0:N2,0:N3), &
      U3NCON(0:N1,0:N2,0:N3))

      ALLOCATE (RO1(0:N1+1,0:N2,0:N3),T1(0:N1+1,0:N2,0:N3), &
      U11(0:N1+1,0:N2,0:N3),U21(0:N1+1,0:N2,0:N3),U31(0:N1+1,0:N2,0:N3),&
      P1(0:N1+1,0:N2,0:N3))

      ALLOCATE (RO2(0:N1,0:N2+1,0:N3),T2(0:N1,0:N2+1,0:N3), &
      U12(0:N1,0:N2+1,0:N3),U22(0:N1,0:N2+1,0:N3),U32(0:N1,0:N2+1,0:N3), &
      P2(0:N1,0:N2+1,0:N3))

      ALLOCATE (RO3(0:N1,0:N2,0:N3+1),T3(0:N1,0:N2,0:N3+1), &
      U13(0:N1,0:N2,0:N3+1),U23(0:N1,0:N2,0:N3+1),U33(0:N1,0:N2,0:N3+1), &
      P3(0:N1,0:N2,0:N3+1))

      ALLOCATE ( F1(0:N1,0:N2,0:N3),F2(0:N1,0:N2,0:N3), &
      F3(0:N1,0:N2,0:N3))
      
      ALLOCATE (Q1(0:N1,0:N2,0:N3), Q2(0:N1,0:N2,0:N3), &
      Q3(0:N1,0:N2,0:N3))

      NMAX=MAX(N1,N2,N3)
 

      ALLOCATE ( RBUF(1:NMAX+1),QBUF(0:NMAX),TFBUF(0:NMAX+1),TBBUF(0:NMAX+1), &
      U2FBUF(0:NMAX+1),U2BBUF(0:NMAX+1),U3FBUF(0:NMAX+1),U3BBUF(0:NMAX+1))

    
      ALLOCATE (SIGM11(0:N1,0:N2,0:N3), &
      SIGM21(0:N1,0:N2,0:N3),SIGM31(0:N1,0:N2,0:N3), &
      SIGM12(0:N1,0:N2,0:N3),SIGM22(0:N1,0:N2,0:N3), &
      SIGM32(0:N1,0:N2,0:N3),SIGM13(0:N1,0:N2,0:N3), &
      SIGM23(0:N1,0:N2,0:N3),SIGM33(0:N1,0:N2,0:N3))

      END SUBROUTINE INPUT

      SUBROUTINE write_data_mpi
      USE BULK
      USE FORCES
      USE CONSTANT
      implicit none
      include 'mpif.h'
    !  logical :: dir_e
      integer i1,i2,j1,j2,k1,k2,ii,c1,c2,c3,i,j,k
      CHARACTER(LEN = 32) FNAMEC
      REAL*8, allocatable :: UCG(:,:,:,:)
      allocate(ucg(5,N1G-1,N2G-1,N3G-1))
      ucg=0.d0
      i1=(N1G-1)/PX*coords(0)+1
      i2=(N1G-1)/PX*(coords(0)+1)
      j1=(N2G-1)/PY*coords(1)+1
      j2=(N2G-1)/PY*(coords(1)+1)
      k1=(N3G-1)/PZ*coords(2)+1
      k2=(N3G-1)/PZ*(coords(2)+1)
      if(coords(0).eq.PX-1)i2=N1G-1
      if(coords(1).eq.PY-1)j2=N2G-1
      if(coords(2).eq.PZ-1)k2=N3G-1	  
!	  write(*,'(a,6(i3,2x))') 'i1,i2,j1,j2,k1,k2=',i1,i2,j1,j2,k1,k2
      ucg(1,i1:i2,j1:j2,k1:k2)=U1CON(1:i2-i1+1,1:j2-j1+1,1:k2-k1+1)
      ucg(2,i1:i2,j1:j2,k1:k2)=U2CON(1:i2-i1+1,1:j2-j1+1,1:k2-k1+1)
      ucg(3,i1:i2,j1:j2,k1:k2)=U3CON(1:i2-i1+1,1:j2-j1+1,1:k2-k1+1)
      ucg(4,i1:i2,j1:j2,k1:k2)=(ROCON(1:i2-i1+1,1:j2-j1+1,1:k2-k1+1)-RO0G)*SOUND*SOUND
      ucg(5,i1:i2,j1:j2,k1:k2)=TCON(1:i2-i1+1,1:j2-j1+1,1:k2-k1+1)
      if(myid.ne.0)then
        call mpi_sendrecv(ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1),MPI_DOUBLE_PRECISION,0,100+myid,ucg(1:5,i1:i2,j1:j2,k1:k2),5*(N1-1)*(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,MPI_PROC_NULL,0, new_comm, status, ierror)
 !        call mpi_sendrecv(ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1),MPI_DOUBLE_PRECISION,0,1000+myid,ucg(1:5,i1:i2,j1:j2,k1:k2),5*(N1-1)*(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,MPI_PROC_NULL,0,mpi_comm_world, status, ierror)
      else
        do ii=1,numprocs-1
            c1=ii/(PY*PZ)
            c2=(ii-c1*PY*PZ)/PZ
            c3=ii-c1*PY*PZ-c2*PZ
            i1=(N1G-1)/PX*c1+1
            i2=(N1G-1)/PX*(c1+1)
            j1=(N2G-1)/PY*c2+1
            j2=(N2G-1)/PY*(c2+1)
            k1=(N3G-1)/PZ*c3+1
            k2=(N3G-1)/PZ*(c3+1)
            if(c1.eq.PX-1)i2=N1G-1
            if(c2.eq.PY-1)j2=N2G-1
            if(c3.eq.PZ-1)k2=N3G-1
		    write(*,'(a,7(i3,2x))') 'ii,i1,i2,j1,j2,k1,k2=',ii,i1,i2,j1,j2,k1,k2			
            call mpi_sendrecv(ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1),MPI_DOUBLE_PRECISION,MPI_PROC_NULL,0,ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1), MPI_DOUBLE_PRECISION,ii,100+ii, new_comm, status, ierror)
 !           call mpi_sendrecv(ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1),MPI_DOUBLE_PRECISION,MPI_PROC_NULL,0,ucg(1:5,i1:i2,j1:j2,k1:k2),5*(i2-i1+1)*(j2-j1+1)*(k2-k1+1), MPI_DOUBLE_PRECISION,ii,1000+ii, mpi_comm_world, status, ierror)
		enddo
      endif
      call mpi_barrier(new_comm,ierror)
      if(myid.eq.0)then
        WRITE(FNAMEC, '("./DATA/DATA_", I7.7,".DAT")') NSTEP
        OPEN(UNIT=20,FILE=FNAMEC,FORM='FORMATTED',STATUS='REPLACE')
        WRITE(20, '(A)') 'TITLE="OUT"'
        WRITE(20, '(A)') 'VARIABLES="X","Y","Z","U1","U2","U3","PC","TC"'
        WRITE(20, 120) TIME,N1G,N2G,N3G
      120 FORMAT('ZONE T="', F10.4, '", I=', I4, ', J=', I4,  ', K=', I4, ', DATAPACKING=BLOCK', /)
        write(20, '(A)') 'VARLOCATION=([4-8]=CELLCENTERED)'
        write(20, '(A,D16.8)') 'STRANDID=1, SOLUTIONTIME =',TIME
        write(20,121) ((((X1W+(i-1)*(X1E-X1W)/(N1G-1))*DCOS(X2S+(j-1)*(X2N-X2S)/(N2G-1)),i=1,N1G),j=1,N2G),k=1,N3G)
        write(20,121) ((((X1W+(i-1)*(X1E-X1W)/(N1G-1))*DSIN(X2S+(j-1)*(X2N-X2S)/(N2G-1)),i=1,N1G),j=1,N2G),k=1,N3G)		
!        write(20,121) (((X1W+(i-1)*(X1E-X1W)/(N1G-1),i=1,N1G),j=1,N2G),k=1,N3G)
!        write(20,121) (((X2S+(j-1)*(X2N-X2S)/(N2G-1),i=1,N1G),j=1,N2G),k=1,N3G)
        write(20,121) (((X3B+(k-1)*(X3T-X3B)/(N3G-1),i=1,N1G),j=1,N2G),k=1,N3G)
        do ii=1,5
            write(20,121) (((ucg(ii,i,j,k),i=1,N1G-1),j=1,N2G-1),k=1,N3G-1)
        enddo
      121 FORMAT(4(D16.8,2x))
        CLOSE(20, STATUS='KEEP')
      endif
      if(myid.eq.0)print *,  'time,nstep,filename=',time,nstep,FNAMEC
      deallocate(ucg)
      return
      END SUBROUTINE
      
      SUBROUTINE INITIAL_DATE
      USE FORCES
      USE BULK
      USE CONSTANT
	  implicit none

      REAL(8) X2C,X3C !X1C,
      REAL(8) ALFA0,BETA,R0,RAD,ETA,UX,UY

!     ЗАДАНИЕ КОНСЕРВАТИВНЫХ ПЕРЕМЕННЫХ   
      ALFA0= 0.204
      BETA=0.3
      R0=0.05

!     НАЧАЛЬНОЕ ОБНУЛЕНИЕ МАССИВОВ  
      U1CON=U10
      U2CON=U20
      U3CON=U30
      ROCON=RO0G
      TCON=T0

      U1NCON=U1CON
      U2NCON=U2CON
      U3NCON=U3CON
      RONCON=RO0G
      TNCON=TCON

      P3=0.d0
      RO3=RO0G
      U13=U10
      U23=U20
      U33=U30
      T3=T0

      P1=0.d0
      RO1=RO0G
      U11=U10
      U21=U20
      U31=U30
      T1=T0

      P2=0.d0
      RO2=RO0G
      U12=U10
      U22=U20
      U32=U30
      T2=T0

      F1=0.d0
      F2=0.d0
      F3=0.d0
!   ЗАДАНИЕ КОНСЕРВАТИВНЫХ ПЕРЕМЕННЫХ     

!      DO K=1,N3-1
!      DO J=1,N2-1
!      DO I=1,N1-1
!        X1C=0.5*(X1(I+1)+X1(I))-0.5
!        X2C=0.5*(X2(J+1)+X2(J))-0.5
!        X3C=0.5*(X3(K+1)+X3(K))-0.5
        
!        RAD=DSQRT(X2C**2+X3C**2)
!        ETA=RAD/R0
!        UFI=ALFA0*ETA*DEXP(BETA*(1.-ETA**2))
!        UY=-UFI*X3C/RAD
!        UZ=UFI*X2C/RAD
!        PRESURE=-RO0G*ALFA0**2*DEXP(2*BETA*(1.-ETA**2))/(4.*BETA)        
!        U1CON(I,J,K)=U10
!        U2CON(I,J,K)=U20
!        U3CON(I,J,K)=U30
!        ROCON(I,J,K)=RO0G
!        TCON(I,J,K)=T0  

!        RONCON(I,J,K)=ROCON(I,J,K)
!        U1NCON(I,J,K)=U1CON(I,J,K)
!        U2NCON(I,J,K)=U2CON(I,J,K)
!        U3NCON(I,J,K)=U3CON(I,J,K)     
!        TNCON(I,J,K)=T0   
!      ENDDO
!      ENDDO
!      ENDDO

!     ЗАДАНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ
!     На гранях, перпендикулярных оси X1
     
!      DO K=1,N3-1
!      DO J=1,N2-1
!      DO I=2,N1-1
      
!        X1C=X1(I)-0.5
!        X2C=0.5*(X2(J+1)+X2(J))-0.5
!        X3C=0.5*(X3(K+1)+X3(K))-0.5
!        RAD=DSQRT(X2C**2+X3C**2)
!        ETA=RAD/R0
!        UFI=ALFA0*ETA*DEXP(BETA*(1.-ETA**2))
!        UY=-UFI*X3C/RAD
!        UZ=UFI*X2C/RAD
!        PRESURE=-RO0G*ALFA0**2*DEXP(2*BETA*(1.-ETA**2))/(4.*BETA)
!        RO1(I,J,K)=RO0G
!        U11(I,J,K)=U10
!        U21(I,J,K)=U20
!        U31(I,J,K)=U30
!        T1(I,J,K)=T0  
!        P1(I,J,K)=0.0
!      ENDDO
!        GOTO 99
 !  УСЛОВИЕ ПРИЛИПАНИЯ 
!        I=1
!        RO1(I,J,K)=RO0G
!        U11(I,J,K)=0.
!        U21(I,J,K)=0.
!        U31(I,J,K)=0.
!        T1(I,J,K)=T0  
!        P1(1,J,K)=0.
		
!        I=N1-1
!        RO1(I+1,J,K)=RO0G
!        U11(I+1,J,K)=0.
!        U21(I+1,J,K)=0.
!        U31(I+1,J,K)=0.
!        T1(I+1,J,K)=0.  
!        P1(I+1,J,K)=0.
! 99   CONTINUE    
!      ENDDO
!      ENDDO

!     На гранях, перпендикулярных оси X2

      
!      DO K=1,N3-1
!      DO I=1,N1-1
!      DO J=1,N2
!        X2C=X2(J)-0.5
!        X1C=0.5*(X1(I+1)+X1(I))-0.5
!        X3C=0.5*(X3(K+1)+X3(K))-0.5
!        RAD=DSQRT(X2C**2+X3C**2)
!        ETA=RAD/R0
!        UFI=ALFA0*ETA*DEXP(BETA*(1.-ETA**2))
!        PRESURE=-RO0G*ALFA0**2*DEXP(2*BETA*(1.-ETA**2))/(4.*BETA)
!        UY=-UFI*X3C/RAD
!        UZ=UFI*X2C/RAD
!        RO2(I,J,K)=RO0G
!        U12(I,J,K)=U10
!        U22(I,J,K)=U20
!        U32(I,J,K)=U30
!        T2(I,J,K)=T0 
!        P2(I,J,K)=0.0
!      ENDDO
!      ENDDO
!      ENDDO
     
!     На гранях, перпендикулярных оси X3
 
      
!      DO I=1,N1-1
!      DO J=1,N2-1
!      DO K=1,N3
!        X3C=X3(K)-0.5
!        X1C=0.5*(X1(I+1)+X1(I))-0.5
!        X2C=0.5*(X2(J+1)+X2(J))-0.5
!        RAD=DSQRT(X2C**2+X3C**2)
!        ETA=RAD/R0
!        UFI=ALFA0*ETA*DEXP(BETA*(1.-ETA**2))
!        PRESURE=-RO0G*ALFA0**2*DEXP(2*BETA*(1.-ETA**2))/(4.*BETA)
!        UY=-UFI*X3C/RAD
!        UZ=UFI*X2C/RAD
!        RO3(I,J,K)=RO0G
!        U13(I,J,K)=U10
!        U23(I,J,K)=U20
!        U33(I,J,K)=U30
!        T3(I,J,K)=T0 
!        P3(I,J,K)=0.0
!      ENDDO
!      ENDDO
!      ENDDO
     
!     Начальное обнуление массивов сил

!      DO K=1,N3
!      DO J=1,N2
!      DO I=1,N1

!      F1(I,J,K)=0.
!      F2(I,J,K)=0.
!      F3(I,J,K)=0.

!      ENDDO
!      ENDDO
!      ENDDO

      END SUBROUTINE INITIAL_DATE

      SUBROUTINE TIME_STEP_SIZE
      USE BULK
      USE FORCES
      USE CONSTANT
	  implicit none
	  include 'mpif.h'
	  integer i,j,k
      REAL(8) rU1C, rU2C, rU3C
      REAL(8) rDT, DTTEMP
      dt=1.d8
      DO K=1,N3-1;      DO J=1,N2-1;      DO I=1,N1-1
        DX1=X1(I+1)-X1(I)
        DX2=X2(J+1)-X2(J)
        DX3=X3(K+1)-X3(K)
        X1C=0.5*(X1(I+1)+X1(I))
        X1L=1+(L-1)*(X1C-1)
        U1C=U1CON(I,J,K)
        U2C=U2CON(I,J,K)
        U3C=U3CON(I,J,K)
        rU1C = U1C
        rU2C = U2C
        rU3C = U3C
        DTU1=CFL*DX1/(SOUND+DABS(rU1C))
        DTU2=CFL*X1L*DX2/(SOUND+DABS(rU2C))
        DTU3=CFL*DX3/(SOUND+DABS(rU3C))
        DTU=DMIN1(DTU1,DTU2,DTU3)
        IF(VIS.gt.1.D-16)then
          DTV1=CFL*DX1**2/(2.*VIS)
          DTV2=CFL*(X1L*DX2)**2/(2.*VIS)
          DTV3=CFL*DX3**2/(2.*VIS)
          DTV=DMIN1(DTV1,DTV2,DTV3)
        ELSE
    	  DTV=1.d16
        ENDIF
        DTTEMP=DMIN1(DTU,DTV)
        IF(DTTEMP.LT.DT)DT=DTTEMP
      ENDDO;      ENDDO;      ENDDO
      call mpi_allreduce(dt,dtmin,1,mpi_double_precision,mpi_min,new_comm,ierror)
 !     call mpi_barrier(new_comm,ierror)
      dt=dtmin
      RETURN
      END SUBROUTINE TIME_STEP_SIZE
	  
      SUBROUTINE PHASE1
      USE BULK
      USE FORCES
      USE CONSTANT
      implicit none
      include 'mpif.h'
      integer i,j,k
      REAL(8) ROCN, U1CP, U2CP, U1CN, U2CN, U3CN
      REAL(8) TOTALFLUX, DS,TOTS,DELTAU3
     !     Пересылка массивов
      call mpi_sendrecv(U1NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,1,U1CON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,1,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,2,U2CON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,2,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,3,U3CON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,3,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,4,ROCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,4,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,5,TCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,5,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,6,U1CON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,6,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,7,U2CON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,7,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,8,U3CON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,8,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,9,ROCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,9,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,10,TCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,10,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,11,U1CON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,11,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,12,U2CON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,12,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,13,U3CON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,13,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,14,ROCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,14,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,15,TCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,15,new_comm, status, ierror)
	  
      call mpi_sendrecv(U1NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,16,U1CON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,16,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,17,U2CON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,17,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,18,U3CON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,18,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,19,ROCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,19,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,20,TCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,20,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,21,U1CON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,21,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,22,U2CON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,22,new_comm, status, ierror)	
      call mpi_sendrecv(U3NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,23,U3CON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,23,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,24,ROCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,24,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,25,TCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,25,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,26,U1CON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,26,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,27,U2CON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,27,new_comm, status, ierror)	
      call mpi_sendrecv(U3NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,28,U3CON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,28,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,29,ROCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,29,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,30,TCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,30,new_comm, status, ierror)
	  
      DO K=1,N3-1;        DO J=1,N2-1;     	DO I=1,N1-1
        U1CON(I,J,K)=U1NCON(I,J,K)
        U2CON(I,J,K)=U2NCON(I,J,K)
        U3CON(I,J,K)=U3NCON(I,J,K)
        ROCON(I,J,K)=RONCON(I,J,K)
        TCON(I,J,K)=TNCON(I,J,K)
      ENDDO;  ENDDO; ENDDO
 !     Расчет консервативных переменных на промежуточном временном слое    
 
      DO K=1,N3-1;      DO J=1,N2-1;      DO I=1,N1-1
 !    геометрические характеристики расчетной ячейки
      DX1=X1(I+1)-X1(I) ! пространственный шаг сетки по направлению X1
      DX2=X2(J+1)-X2(J) ! пространственный шаг сетки по направлению X2
      DX3=X3(K+1)-X3(K) ! пространственный шаг сетки по направлению X3

      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности
      XLT=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности
      XLB=XLT                !  геометрический фактор цилиндричности
      
      DVC=0.5*(XLE+XLW)*DX1*DX2*DX3 ! Объем расчетной ячейки
      DS1=DX2*DX3   ! Площадь грани, перпендикулярной X1
      DS2=DX1*DX3   ! Площадь грани, перпендикулярной X2
      DS3=DX1*DX2   ! Площадь грани, перпендикулярной X3
      ALFA=DT*(L-1)*DX1*DX2*DX3/(2*DVC) ! геметрический параметр

 !    загрузка из массивов рабочих переменных
 !    для граней ячеек (потоковые переменные)
  !    Давления
      PE=P1(I+1,J,K)
      PW=P1(I,J,K)
      PN=P2(I,J+1,K)
      PS=P2(I,J,K)
      PT=P3(I,J,K+1)
      PB=P3(I,J,K)
      
  !    Компоненты скорости, плотность и температура    
      U1E=U11(I+1,J,K)
      U2E=U21(I+1,J,K)
      U3E=U31(I+1,J,K)
      ROE=RO1(I+1,J,K)
      TE=T1(I+1,J,K)
      
      U1W=U11(I,J,K)
      U2W=U21(I,J,K)
      U3W=U31(I,J,K)
      ROW=RO1(I,J,K)
      TW=T1(I,J,K)
      
      U1N=U12(I,J+1,K)
      U2N=U22(I,J+1,K)
      U3N=U32(I,J+1,K)
      RON=RO2(I,J+1,K)
      TN=T2(I,J+1,K)
      
      U1S=U12(I,J,K)
      U2S=U22(I,J,K)
      U3S=U32(I,J,K)
      ROS=RO2(I,J,K)
      TS=T2(I,J,K)
      
      U1T=U13(I,J,K+1)
      U2T=U23(I,J,K+1)
      U3T=U33(I,J,K+1)
      ROT=RO3(I,J,K+1)
      TT=T3(I,J,K+1)
      
      U1B=U13(I,J,K)
      U2B=U23(I,J,K)
      U3B=U33(I,J,K)
      ROB=RO3(I,J,K)
      TB=T3(I,J,K)
      
!    для центров ячеек (консервативные переменные)

      U1C=U1CON(I,J,K)
      U2C=U2CON(I,J,K)
      U3C=U3CON(I,J,K)
      ROC=ROCON(I,J,K)
      TC=TCON(I,J,K)
   
 !    Вычисление консервативных переменных на полуслое
 
 !    Новые плотности 
      ROCN=(ROC*DVC-0.5*DT*((XLE*ROE*U1E-XLW*ROW*U1W)*DS1+(RON*U2N-ROS*U2S)*DS2 &
      +(XLT*ROT*U3T-XLB*ROB*U3B)*DS3))/DVC  
     
  !    Новые консервативные скорости по оси X1   
      U1CP=(ROC*U1C*DVC-0.5*DT*((RON*U2N*U1N-ROS*U2S*U1S)*DS2+ &
      (XLT*ROT*U3T*U1T-XLB*ROB*U3B*U1B)*DS3+ &
      (XLE*(ROE*U1E**2+PE)-XLW*(ROW*U1W**2+PW))*DS1 &
      -0.5*(L-1)*(PE+PW)*DX1*DX2*DX3))/(DVC*ROCN)
           
 !    Новые консервативные скорости по оси X2  
 
      U2CP=(ROC*U2C*DVC-0.5*DT*((XLE*ROE*U1E*U2E-XLW*ROW*U1W*U2W)*DS1+ &
     ((RON*U2N**2+PN)-(ROS*U2S**2+PS))*DS2+ &
      (XLT*ROT*U3T*U2T-XLB*ROB*U3B*U2B)*DS3))/(DVC*ROCN)    
     
 !    Учет центробежных и кориолисовых сил
 
      U1CN=(U1CP-ALFA*U2C*U1CP)/(1+(ALFA*U2C)**2)
      U2CN=U2CP-ALFA*U2C*U1CN
     
 !    Новые консервативные скорости по оси X3
 
      U3CN=(ROC*U3C*DVC-0.5*DT*((XLE*ROE*U1E*U3E-XLW*ROW*U1W*U3W)*DS1+ &
     (RON*U2N*U3N-ROS*U2S*U3S)*DS2+&
     (XLT*(ROT*U3T**2+PT)-XLB*(ROB*U3B**2+PB))*DS3))/(DVC*ROCN) 
     
 !    Новые температуры
 
       TCN=(ROC*TC*DVC-0.5*DT*((XLE*ROE*TE*U1E-XLW*ROW*TW*U1W)*DS1+&
      (RON*TN*U2N-ROS*TS*U2S)*DS2+&
      (XLT*ROT*TT*U3T-XLB*ROB*TB*U3B)*DS3))/(DVC*ROCN) 
      

        TNCON(I,J,K)=TCN  
        RONCON(I,J,K)=ROCN
        U1NCON(I,J,K)=U1CN
        U2NCON(I,J,K)=U2CN
        U3NCON(I,J,K)=U3CN
     
      ENDDO;      ENDDO;      ENDDO
!   ПЕРИОДИЧНОСТЬ 

!      DO K=1,N3;      DO I=1,N1
!      RONCON(I,N2,K)=RONCON(I,1,K)
!      U1NCON(I,N2,K)=U1NCON(I,1,K)
!      U2NCON(I,N2,K)=U2NCON(I,1,K)
!      U3NCON(I,N2,K)=U3NCON(I,1,K)
!      TNCON(I,N2,K)=TNCON(I,1,K)
      
!      RONCON(I,0,K)=RONCON(I,N2-1,K)
!      U1NCON(I,0,K)=U1NCON(I,N2-1,K)
!      U2NCON(I,0,K)=U2NCON(I,N2-1,K)
!      U3NCON(I,0,K)=U3NCON(I,N2-1,K)
!      TNCON(I,0,K)=TNCON(I,N2-1,K)
!      ENDDO;      ENDDO

!   ПРИЛИПАНИЕ
      if(coords(0).eq.0)then
        DO K=1,N3-1;      DO J=1,N2-1
          RONCON(0,J,K)=RONCON(1,J,K)
          U1NCON(0,J,K)=0.
          U2NCON(0,J,K)=0.
          U3NCON(0,J,K)=0.
          TNCON(0,J,K)= T0
        ENDDO;      ENDDO
      endif
      if(coords(0).eq.px-1)then
        DO K=1,N3-1;      DO J=1,N2-1	  
          RONCON(N1,J,K)=RONCON(N1-1,J,K)
          U1NCON(N1,J,K)=0.
          U2NCON(N1,J,K)=0.
          U3NCON(N1,J,K)=0.
          TNCON(N1,J,K)= T0
        ENDDO;      ENDDO
      endif
 !     DO K=1,N3;      DO J=1,N2
 !     RONCON(N1,J,K)=RONCON(N1-1,J,K)
 !     U1NCON(N1,J,K)=0.
 !     U2NCON(N1,J,K)=0.
 !     U3NCON(N1,J,K)=0.
 !     TNCON(N1,J,K)= T0
      
 !     RONCON(0,J,K)=RONCON(1,J,K)
 !     U1NCON(0,J,K)=0.
 !     U2NCON(0,J,K)=0.
 !     U3NCON(0,J,K)=0.
 !     TNCON(0,J,K)= T0
 !     ENDDO;      ENDDO
 
      return	  
      END SUBROUTINE PHASE1
	  
      subroutine NEWT
      use BULK
      use CONSTANT
      use FORCES
      implicit none
      include 'mpif.h'	  
      integer i,j,k	  
      do k=1,N3-1
          FQ = 0 !4*Kappa*(PI**2)*sin(2.*PI*(X3(k)+DX3/2))
          do j=1,N2-1
              do i=1,N1-1
                 TNCON(i,j,k) = TNCON(i,j,k)- &
                     (DT/2)*((Q1(i+1,j,k)-Q1(i,j,k))/(X1(i+1)-X1(i)) + &
			                 (Q2(i,j+1,k)-Q2(i,j,k))/(X2(j+1)-X2(j)) + &
			                 (Q3(i,j,k+1)-Q3(i,j,k))/(X3(k+1)-X3(k)) + FQ)
              enddo
          enddo
      enddo
      end subroutine NEWT      

      SUBROUTINE STRESSTENSOR
!     ВЫЧИСЛЕНИЕ КОМПОНЕНТ ТЕНЗОРА ВЯЗКИХ НАПРЯЖЕНИЙ
      USE BULK
      USE FORCES
      USE CONSTANT
      USE STRESS
      implicit none
      include 'mpif.h'	  
      integer i,j,k,k1,k2
      real*8 DX2S,DX2N,DX3B,DX3T,XLN,U1CE,U2CE,U3CE,U1CT,U2CT,U3CT,SIGM1C,SIGM2C
!     Обнуление массивов компонент тензора напряжений
      SIGM11=0.d0
      SIGM21=0.d0
      SIGM31=0.d0

      SIGM12=0.d0
      SIGM22=0.d0
      SIGM32=0.d0

      SIGM13=0.d0
      SIGM23=0.d0
      SIGM33=0.d0
!     ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ ОКАЙМЛЕНИЕМ МАССИВОВ
!     ПО ГРАНИЦАМ, ПЕРПЕНДИКУЛЯРНЫМ X1 ЗАДАЕТСЯ УСЛОВИЕ ПРИЛИПАНИЯ
      X1(0)=2*X1(1)-X1(2)
      X1(N1+1)=2*X1(N1)-X1(N1-1)
      if(coords(0).eq.0)then
         X1(0)=X1(1)
         U1CON(0,1:N2-1,1:N3-1)=0.d0
         U2CON(0,1:N2-1,1:N3-1)=0.d0
         U3CON(0,1:N2-1,1:N3-1)=0.d0	  
      endif
      if(coords(0).eq.px-1)then
         X1(N1+1)=X1(N1)
         U1CON(N1,1:N2-1,1:N3-1)=0.d0
         U2CON(N1,1:N2-1,1:N3-1)=0.d0
         U3CON(N1,1:N2-1,1:N3-1)=0.d0
      endif
!     ПО ГРАНИЦАМ, ПЕРПЕНДИКУЛЯРНЫМ X2 ЗАДАЕТСЯ УСЛОВИЕ ПЕРИОДИЧНОСТИ
      DX2S=X2(2)-X2(1)
      DX2N=X2(N2)-X2(N2-1)
      X2(0)=X2(1)-DX2N
      X2(N2+1)=X2(N2)+DX2S
 !     DO I=1,N1-1
 !     DO K=1,N3-1

 !     U1CON(I,0,K)=U1CON(I,N2-1,K)
 !     U1CON(I,N2,K)=U1CON(I,1,K)

 !     U2CON(I,0,K)=U2CON(I,N2-1,K)
 !     U2CON(I,N2,K)=U2CON(I,1,K)

 !     U3CON(I,0,K)=U3CON(I,N2-1,K)
 !     U3CON(I,N2,K)=U3CON(I,1,K)


 !     ENDDO
 !     ENDDO
!     ПО ГРАНИЦАМ, ПЕРПЕНДИКУЛЯРНЫМ X3 ЗАДАЕТСЯ УСЛОВИЕ ПЕРИОДИЧНОСТИ
      DX3B=X3(2)-X3(1)
      DX3T=X3(N3)-X3(N3-1)
      X3(0)=X3(1)-DX3B
      X3(N3+1)=X3(N3)+DX3T

!     DO I=1,N1-1
!     DO J=1,N2-1

!      U1CON(I,J,0)=U1CON(I,J,N3-1)
!      U1CON(I,J,N3)=U1CON(I,J,1)

!      U2CON(I,J,0)=U2CON(I,J,N3-1)
!      U2CON(I,J,N3)=U2CON(I,J,1)

!      U3CON(I,J,0)=U3CON(I,J,N3-1)
!      U3CON(I,J,N3)=U3CON(I,J,1)


!      ENDDO
!      ENDDO

!     ОБХОД ПО ГРАНЯМ
!     Обход по граням, перпендикулярным X1
!   ВЫЧИСЛЕНИЕ КОМПОНЕНТ ТЕНЗОРА НА ГРАНЯХ, ПЕРПЕНДИКУЛЯРНЫХ НАПРАВЛЕНИЮ 1
      DO K=1,N3-1

      DX3=X3(K+1)-X3(K) ! пространственный шаг сетки по направлению X3

      DO J=1,N2-1
      
      DX2=X2(J+1)-X2(J) ! пространственный шаг сетки по направлению X2
      DS1=DX2*DX3   ! Площадь грани, перпендикулярной X1

      DO I=1,N1

    
      DX1=0.5*(X1(I+1)-X1(I-1))
     
!    геометрические характеристики расчетной ячейки
      XLE=1+(L-1)*(X1(I)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I-1)-1)  !  геометрический фактор цилиндричности   
      XLT=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности   
      XLN=XLT                !  геометрический фактор цилиндричности
      
!    Компоненты скорости в центрах ячеек   

      U1C=U1CON(I,J,K)
      U1CE=U1CON(I-1,J,K)
       
      U2C=U2CON(I,J,K)
      U2CE=U2CON(I-1,J,K)
       
      U3C=U3CON(I,J,K)
      U3CE=U3CON(I-1,J,K)
       
!    Компоненты тензора вязких напряжений
!     SIGMA11 - напряжение трения в направлении X1 для грани, перпендикулярной оси X1
!     SIGMA21 - напряжение трения в направлении X2 для грани, перпендикулярной оси X1
!     SIGMA31 - напряжение трения в направлении X3 для грани, перпендикулярной оси X1  
 
      SIGM11(I,J,K)=-VIS*XLE*(U1CE-U1C)/DX1
      SIGM21(I,J,K)=-VIS*XLE*(U2CE-U2C)/DX1 
      SIGM31(I,J,K)=-VIS*XLE*(U3CE-U3C)/DX1    
        
      ENDDO
      ENDDO
      ENDDO
!     Обход по граням, перпендикулярным X2
 
      DO K=1,N3-1;      DO I=1,N1-1

      DX3=X3(K+1)-X3(K) ! пространственный шаг сетки по направлению X3
      DX1=X1(I+1)-X1(I) ! пространственный шаг сетки по направлению X1
      DS2=DX1*DX3   ! Площадь грани, перпендикулярной X2

!    геометрические характеристики расчетной ячейки
      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности   
      XLT=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности   
      XLN=XLT                !  геометрический фактор цилиндричности

!     ПЕРИОДИЧЕСКИЕ ГРАНИЧНЫЕ УСЛОВИЯ НА СЕВЕРНОЙ И ЮЖНОЙ ГРАНИЦАХ

      DO J=1,N2
      
 
      DX2=0.5*(X2(J+1)-X2(J-1)) ! пространственный шаг сетки по направлению X2  
      
 !    Компоненты скорости в центрах ячеек   
  
      U1C=U1CON(I,J,K) 
      U1CN=U1CON(I,J-1,K)
       
      
      U2C=U2CON(I,J,K) 
      U2CN=U2CON(I,J-1,K)
     
      
      U3C=U3CON(I,J,K) 
      U3CN=U3CON(I,J-1,K)
       
      
!    Компоненты тензора вязких напряжений
 
!     SIGMA12 - напряжение трения в направлении X1 для грани, перпендикулярной оси X2
!     SIGMA22 - напряжение трения в направлении X2 для грани, перпендикулярной оси X2
!     SIGMA32 - напряжение трения в направлении X3 для грани, перпендикулярной оси X2
   
      SIGM12(I,J,K)=-VIS*((U1CN-U1C)/DX2 -(L-1)*(U2C+U2CN))/XLN  
      SIGM22(I,J,K)=-VIS*((U2CN-U2C)/DX2 +(L-1)*(U1C+U1CN))/XLN  
      SIGM32(I,J,K)=-VIS*(U3CN-U3C)/DX2
       
      ENDDO
      ENDDO;      ENDDO
 
!     Обход по граням, перпендикулярным X3
 
      DO I=1,N1-1;      DO J=1,N2-1

!     ГРАНИЧНЫЕ УСЛОВИЯ НА НИЖНЕЙ И ВЕРХНЕЙ ГРАНИЦАХ НЕ ДАЮТ ВКЛАДА В СИЛЫ F1,F2,F3

      DX1=X1(I+1)-X1(I) ! пространственный шаг сетки по направлению X1
      DX2=X2(J+1)-X2(J) ! пространственный шаг сетки по направлению X2
!    геометрические характеристики расчетной ячейки
      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности   
      XLT=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности   
      XLN=XLT                !  геометрический фактор цилиндричности
      DS3=DX1*DX2   ! Площадь грани, перпендикулярной X3

      k1=1
      k2=N3
      if(coords(2).eq.0)k1=2
      if(coords(2).eq.pz-1)k2=N3-1
      DO K=k1,k2
      DX3=0.5*(X3(K+1)-X3(K-1))
 !    Компоненты скорости в центрах ячеек    
      U1C=U1CON(I,J,K) 
      U1CT=U1CON(I,J,K-1)
      
      U2C=U2CON(I,J,K)
      U2CT=U2CON(I,J,K-1)
      
      U3C=U3CON(I,J,K) 
      U3CT=U3CON(I,J,K-1)   
!    Компоненты тензора вязких напряжений
 
!     SIGMA13 - напряжение трения в направлении X1 для грани, перпендикулярной оси X3
!     SIGMA23 - напряжение трения в направлении X2 для грани, перпендикулярной оси X3
!     SIGMA33 - напряжение трения в направлении X3 для грани, перпендикулярной оси X3
    
      SIGM13(I,J,K)=-VIS*XLT*(U1CT-U1C)/DX3
      SIGM23(I,J,K)=-VIS*XLT*(U2CT-U2C)/DX3 
      SIGM33(I,J,K)=-VIS*XLT*(U3CT-U3C)/DX3    
      ENDDO
      ENDDO;      ENDDO

!     ВЫЧИСЛЕНИЕ СИЛ ТРЕНИЯ

      DO I=1,N1-1;      DO J=1,N2-1;      DO K=1,N3-1

      DX1=X1(I+1)-X1(I)
      DX2=X2(J+1)-X2(J)
      DX3=X3(K+1)-X3(K)

      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности   
      XLT=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности   

      DS1=DX2*DX3
      DS2=DX1*DX3
      DS3=DX1*DX2

      SIGM1C=VIS*U1CON(I,J,K)/XLT
      SIGM2C=VIS*U2CON(I,J,K)/XLT

      F1(I,J,K)=(SIGM11(I+1,J,K)-SIGM11(I,J,K))*DS1+ &
      (SIGM12(I,J+1,K)-SIGM12(I,J,K))*DS2+ &
      (SIGM13(I,J,K+1)-SIGM13(I,J,K))*DS3-(L-1)*SIGM1C*DX1*DX2*DX3

      F2(I,J,K)=(SIGM21(I+1,J,K)-SIGM21(I,J,K))*DS1+ &
      (SIGM22(I,J+1,K)-SIGM22(I,J,K))*DS2+ &
      (SIGM23(I,J,K+1)-SIGM23(I,J,K))*DS3-(L-1)*SIGM2C*DX1*DX2*DX3

      F3(I,J,K)=(SIGM31(I+1,J,K)-SIGM31(I,J,K))*DS1+ &
      (SIGM32(I,J+1,K)-SIGM32(I,J,K))*DS2+ &
      (SIGM33(I,J,K+1)-SIGM33(I,J,K))*DS3


      ENDDO;      ENDDO;      ENDDO
      RETURN
      END SUBROUTINE STRESSTENSOR
	  
      SUBROUTINE USEFORCES
      USE FORCES
      USE BULK
      USE CONSTANT
      implicit none
      include 'mpif.h'
      integer i,j,k  

      DO K=1,N3-1
      DO J=1,N2-1

      DX2=X2(J+1)-X2(J) ! пространственный шаг сетки по направлению X2
      DX3=X3(K+1)-X3(K) ! пространственный шаг сетки по направлению X3

      DO I=1,N1-1

      DX1=X1(I+1)-X1(I) ! пространственный шаг сетки по направлению X1
!    геометрические характеристики расчетной ячейки
      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности   
      DVC=0.5*(XLE+XLW)*DX1*DX2*DX3 ! Объем расчетной ячейки

      ROC=ROCON(I,J,K)
      ROCN=RONCON(I,J,K)
 
      U1NCON(I,J,K)=(ROC*DVC*U1NCON(I,J,K)+0.5*DT*F1(I,J,K))/(DVC*ROCN)
      U2NCON(I,J,K)=(ROC*DVC*U2NCON(I,J,K)+0.5*DT*F2(I,J,K))/(DVC*ROCN)
      U3NCON(I,J,K)=(ROC*DVC*U3NCON(I,J,K)+0.5*DT*F3(I,J,K))/(DVC*ROCN)

      ENDDO
      ENDDO
      ENDDO
      END SUBROUTINE USEFORCES
	  
      SUBROUTINE PHASE2
      USE BULK
      USE FORCES
      USE CONSTANT
      implicit none
      include 'mpif.h'
      integer i,j,k,i1,i2,j1,j2,k1,k2
      REAL(8) RF,RC,RB,QF,QC,TF,TC,TB,U2F,U2C,U2B,U3F,U3B,U3C,U1F,U1C,U1B
      !передача всех необходимых массивов
   
      call mpi_sendrecv(U11(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,31,U11(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,31,new_comm, status, ierror)
      call mpi_sendrecv(U21(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,32,U21(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,32,new_comm, status, ierror)
      call mpi_sendrecv(U31(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,33,U31(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,33,new_comm, status, ierror)
      call mpi_sendrecv(RO1(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,34,RO1(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,34,new_comm, status, ierror)
      call mpi_sendrecv(T1(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,35,T1(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,35,new_comm, status, ierror)      
      call mpi_sendrecv(P1(2,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,36,P1(N1+1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,36,new_comm, status, ierror)	  

      call mpi_sendrecv(U11(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,37,U11(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,37,new_comm, status, ierror)
      call mpi_sendrecv(U21(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,38,U21(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,38,new_comm, status, ierror)
      call mpi_sendrecv(U31(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,39,U31(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,39,new_comm, status, ierror)
      call mpi_sendrecv(RO1(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,40,RO1(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,40,new_comm, status, ierror)
      call mpi_sendrecv(T1(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,41,T1(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,41,new_comm, status, ierror)	  
      call mpi_sendrecv(P1(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,42,P1(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,42,new_comm, status, ierror)

      call mpi_sendrecv(U12(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,43,U12(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,43,new_comm, status, ierror)
      call mpi_sendrecv(U22(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,44,U22(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,44,new_comm, status, ierror)
      call mpi_sendrecv(U32(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,45,U32(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,45,new_comm, status, ierror)
      call mpi_sendrecv(RO2(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,46,RO2(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,46,new_comm, status, ierror)
      call mpi_sendrecv(T2(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,47,T2(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,47,new_comm, status, ierror)
      call mpi_sendrecv(P2(1:N1-1,2,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,48,P2(1:N1-1,N2+1,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,48,new_comm, status, ierror)
	  
      call mpi_sendrecv(U12(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,49,U12(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,49,new_comm, status, ierror)
      call mpi_sendrecv(U22(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,50,U22(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,50,new_comm, status, ierror)
      call mpi_sendrecv(U32(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,51,U32(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,51,new_comm, status, ierror)
      call mpi_sendrecv(RO2(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,52,RO2(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,52,new_comm, status, ierror)
      call mpi_sendrecv(T2(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,53,T2(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,53,new_comm, status, ierror)
      call mpi_sendrecv(P2(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,54,P2(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,54,new_comm, status, ierror)
	  
      call mpi_sendrecv(U13(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,55,U13(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,55,new_comm, status, ierror)
      call mpi_sendrecv(U23(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,56,U23(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,56,new_comm, status, ierror)	
      call mpi_sendrecv(U33(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,57,U33(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,57,new_comm, status, ierror)
      call mpi_sendrecv(RO3(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,58,RO3(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,58,new_comm, status, ierror)
      call mpi_sendrecv(T3(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,59,T3(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,59,new_comm, status, ierror)
      call mpi_sendrecv(P3(1:N1-1,1:N2-1,2),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,60,P3(1:N1-1,1:N2-1,N3+1),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,60,new_comm, status, ierror)
	  
      call mpi_sendrecv(U13(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,61,U13(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,61,new_comm, status, ierror)
      call mpi_sendrecv(U23(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,62,U23(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,62,new_comm, status, ierror)	
      call mpi_sendrecv(U33(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,63,U33(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,63,new_comm, status, ierror)
      call mpi_sendrecv(RO3(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,64,RO3(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,64,new_comm, status, ierror)
      call mpi_sendrecv(T3(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,65,T3(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,65,new_comm, status, ierror)	  
      call mpi_sendrecv(P3(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,66,P3(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,66,new_comm, status, ierror)
 
       call mpi_sendrecv(U1NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,67,U1NCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,67,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,68,U2NCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,68,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,69,U3NCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,69,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,70,RONCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,70,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,left,71,TNCON(N1,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,right,71,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,72,U1NCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,72,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,73,U2NCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,73,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,74,U3NCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,74,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,75,RONCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,75,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(N1-1,1:N2-1,1:N3-1),(N2-1)*(N3-1),MPI_DOUBLE_PRECISION,right,76,TNCON(0,1:N2-1,1:N3-1),(N2-1)*(N3-1), MPI_DOUBLE_PRECISION,left,76,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,77,U1NCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,77,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,78,U2NCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,78,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,79,U3NCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,79,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,80,RONCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,80,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,down,81,TNCON(1:N1-1,N2,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,up,81,new_comm, status, ierror)
	  
      call mpi_sendrecv(U1NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,82,U1NCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,82,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,83,U2NCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,83,new_comm, status, ierror)
      call mpi_sendrecv(U3NCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,84,U3NCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,84,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,85,RONCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,85,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,N2-1,1:N3-1),(N1-1)*(N3-1),MPI_DOUBLE_PRECISION,up,86,TNCON(1:N1-1,0,1:N3-1),(N1-1)*(N3-1), MPI_DOUBLE_PRECISION,down,86,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,87,U1NCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,87,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,88,U2NCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,88,new_comm, status, ierror)	
      call mpi_sendrecv(U3NCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,89,U3NCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,89,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,90,RONCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,90,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1:N2-1,1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,near,91,TNCON(1:N1-1,1:N2-1,N3),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,far,91,new_comm, status, ierror)

      call mpi_sendrecv(U1NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,92,U1NCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,92,new_comm, status, ierror)
      call mpi_sendrecv(U2NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,93,U2NCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,93,new_comm, status, ierror)	
      call mpi_sendrecv(U3NCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,94,U3NCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,94,new_comm, status, ierror)
      call mpi_sendrecv(RONCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,95,RONCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,95,new_comm, status, ierror)
      call mpi_sendrecv(TNCON(1:N1-1,1:N2-1,N3-1),(N1-1)*(N2-1),MPI_DOUBLE_PRECISION,far,96,TNCON(1:N1-1,1:N2-1,0),(N1-1)*(N2-1), MPI_DOUBLE_PRECISION,near,96,new_comm, status, ierror) 
 !     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ГРАНЯХ DS1, ортогональных оси X1
      
     
!   ПРОХОД ПО НАПРАВЛЕНИЮ 1 (X,r)   
!   ВНАЧАЛЕ ЛОКАЛЬНЫЕ ИНВАРИАНТЫ ДЛЯ ВНУТРЕННИХ ГРАНЕЙ ЗАНОСЯТСЯ В БУФЕРНЫЕ МАССИВЫ, ОБХОД ПРОИЗВОДИТСЯ ПО ЦЕТРАМ ЯЧЕЕК,
!   ЗАТЕМ С УЧЕТОМ ГРАНИЧНЫХ УСЛОВИЙ ВЫЧИСЛЯЮТСЯ КРАЙНИЕ ЭЛЕМЕНТЫ БУФЕРОВ, И ТОЛьКО ЗАТЕМ ВЫЧИСЛЯЮТСЯ ПОТОКОВЫЕ ПЕРЕМЕННЫЕ. 

      DO K=1,N3-1  ! ТОЛЬКО ДЛЯ ВНУТРЕННИХ ГРАНЕЙ
      DO J=1,N2-1
      i1=0
      i2=N1
      if(coords(0).eq.0)i1=1
      if(coords(0).eq.px-1)i2=N1-1
      DO i=i1,i2

      DX=X1(I+1)-X1(I)
 
      UF=U11(I+1,J,K)
      UB=U11(I,J,K)
      UCN=U1NCON(I,J,K)
      UC=U1CON(I,J,K)
       
      U2F=U21(I+1,J,K)
      U2B=U21(I,J,K)
      U2CN=U2NCON(I,J,K)
      U2C=U2CON(I,J,K)
       
      U3F=U31(I+1,J,K)  ! !!!!!!!!!!!!!!!!!!!!!!!!!!!
      U3B=U31(I,J,K)    ! !!!!!!!!!!!!!!!!!!!!!!!!!!
      U3CN=U3NCON(I,J,K)
      U3C=U3CON(I,J,K)
      
!      RO0=RO0G  !!DNINT(RONCON(I,J,K))
       
      ROF=RO1(I+1,J,K)
      ROB=RO1(I,J,K)
      ROCN=RONCON(I,J,K)
      ROC=ROCON(I,J,K)

      TF=T1(I+1,J,K)
      TB=T1(I,J,K)
      TCN=TNCON(I,J,K)
      TC=TCON(I,J,K)
       
     
      PF=P1(I+1,J,K)
      PB=P1(I,J,K)
      PCN=SOUND**2*(ROCN-RO0G)
      PC=SOUND**2*(ROC-RO0G)

!     Вычисление инвариантов

      RF=UF+PF/(RO0G*SOUND)
      RB=UB+PB/(RO0G*SOUND)
      RCN=UCN+PCN/(RO0G*SOUND)
      RC=UC+PC/(RO0G*SOUND)
        
      RFN=2*RCN-RB
     
      QF=UF-PF/(RO0G*SOUND)
      QB=UB-PB/(RO0G*SOUND)
      QCN=UCN-PCN/(RO0G*SOUND)
      QC=UC-PC/(RO0G*SOUND)
           
      QBN=2*QCN-QF

      TFN=2*TCN-TB
      TBN=2*TCN-TF

      U2FN=2*U2CN-U2B
      U2BN=2*U2CN-U2F

      U3FN=2*U3CN-U3B 
      U3BN=2*U3CN-U3F 

!     Допустимый диапазон их изменения


      GR=2*(RCN-RC)/DT+(UCN+SOUND)*(RF-RB)/DX
      GQ=2*(QCN-QC)/DT+(UCN-SOUND)*(QF-QB)/DX

      GT=2*(TCN-TC)/DT+UCN*(TF-TB)/DX
      GU2=2*(U2CN-U2C)/DT+UCN*(U2F-U2B)/DX
      GU3=2*(U3CN-U3C)/DT+UCN*(U3F-U3B)/DX  
      
      RMAX=DMAX1(RF,RC,RB) +DT*GR
      RMIN=DMIN1(RF,RC,RB) +DT*GR

      QMAX=DMAX1(QF,QC,QB) +DT*GQ
      QMIN=DMIN1(QF,QC,QB) +DT*GQ
 
 
      TMAX=DMAX1(TF,TC,TB) +DT*GT
      TMIN=DMIN1(TF,TC,TB) +DT*GT
      
      U2MAX=DMAX1(U2F,U2C,U2B) +DT*GU2
      U2MIN=DMIN1(U2F,U2C,U2B) +DT*GU2

      U3MAX=DMAX1(U3F,U3C,U3B) +DT*GU3  
      U3MIN=DMIN1(U3F,U3C,U3B) +DT*GU3  

!     Коррекция инвариантов

      IF(RFN.GT.RMAX) RFN=RMAX
      IF(RFN.LT.RMIN) RFN=RMIN

      IF(QBN.GT.QMAX) QBN=QMAX
      IF(QBN.LT.QMIN) QBN=QMIN

      IF(TFN.GT.TMAX) TFN=TMAX
      IF(TFN.LT.TMIN) TFN=TMIN

      IF(TBN.GT.TMAX) TBN=TMAX
      IF(TBN.LT.TMIN) TBN=TMIN

      IF(U2FN.GT.U2MAX) U2FN=U2MAX
      IF(U2FN.LT.U2MIN) U2FN=U2MIN

      IF(U2BN.GT.U2MAX) U2BN=U2MAX
      IF(U2BN.LT.U2MIN) U2BN=U2MIN

      IF(U3FN.GT.U3MAX) U3FN=U3MAX  
      IF(U3FN.LT.U3MIN) U3FN=U3MIN  

      IF(U3BN.GT.U3MAX) U3BN=U3MAX  
      IF(U3BN.LT.U3MIN) U3BN=U3MIN  

!     Засылка инвариантов в буферные массивы

      RBUF(I+1)=RFN
      QBUF(I)=QBN
      TFBUF(I+1) =TFN
      TBBUF(I)=TBN
      U2FBUF(I+1)=U2FN
      U2BBUF(I)=U2BN
      U3FBUF(I+1)=U3FN  
      U3BBUF(I)=U3BN    

      ENDDO

!     ГРАНИЧНЫЕ УСЛОВИЯ ПО НАПРАВЛЕНИЮ 1 (X,r)
!     ЗАДАНИЕ ГРАНИЧНЫХ ИНВАРИАНТОВ И ЗАНЕСЕНИЕ ИХ В БУФЕРНЫЕ МАССИВЫ 

!      RBUF(1)=RBUF(N1)      !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TFBUF(1)=TFBUF(N1)
!      U2FBUF(1)=U2FBUF(N1)
!      U3FBUF(1)= U3FBUF(N1)

!      QBUF(N1)=QBUF(1)       !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TBBUF(N1)= TBBUF(1)
!      U2BBUF(N1)=U2BBUF(1)
!      U3BBUF(N1)=U3BBUF(1)    
 


!        GOTO 88  
         
!     ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ ПРИЛИПАНИЯ
  
!     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ГРАНИЦАХ БУФЕРНОЙ СТРОКИ
      if(coords(0).eq.0)then
      I=1
      
      RO0B=RONCON(N1-1,J,K)
      RO0F=RONCON(1,J,K)
          
 
      QN=QBUF(I)

      UN=0.
      PN=-QN*SOUND*RO0G       
      RON=(RO0G+PN/SOUND**2)  

  
      TN=T0
      U2N=0.
      U3N=0.  
    
  
      
      P1(I,J,K)=PN
      U11(I,J,K)=UN
      RO1(I,J,K)=RON
      T1(I,J,K)=TN
      U21(I,J,K)=U2N
      U31(I,J,K)=U3N  
      endif
      if(coords(0).eq.px-1)then
      I=N1

      RN=RBUF(I)
 

      UN=0.
      PN=RN*SOUND*RO0G          
      RON=(RO0G+PN/SOUND**2)

 

      TN=T0
      U2N=0
      U3N=0
      
 
      P1(I,J,K)=PN
      U11(I,J,K)=UN
      RO1(I,J,K)=RON
      T1(I,J,K)=TN
      U21(I,J,K)=U2N
      U31(I,J,K)=U3N
      endif
!     Вычисление потоковых переменных  
88     CONTINUE
      i1=1
      i2=N1
      if(coords(0).eq.0)i1=2
      if(coords(0).eq.px-1)i2=N1-1
      DO i=i1,i2

      RO0B=RONCON(I-1,J,K)
      RO0F=RONCON(I,J,K)

      RN=RBUF(I)
      QN=QBUF(I)
    
      PN=(RN-QN)*SOUND*RO0G/2
      UN=(RN+QN)/2
      
      RON=(RO0G+PN/SOUND**2)
      
      UCF=U1NCON(I,J,K)
      UCB=U1NCON(I-1,J,K)
      
      IF(UCF.GE.0.AND.UCB.GT.0) THEN 
      TN=TFBUF(I)
      U2N=U2FBUF(I)
      U3N=U3FBUF(I) 
      ELSE
        IF(UCF.LE.0.AND.UCB.LE.0) THEN   
        TN=TBBUF(I)
        U2N=U2BBUF(I)
        U3N=U3BBUF(I)
        ELSE
        ENDIF
                IF(UCB.GE.0.AND.UCF.LE.0) THEN
                IF(UCB.GT.-UCF) THEN
                TN=TFBUF(I) 
                U2N=U2FBUF(I)
                U3N=U3FBUF(I)  
                ELSE
                TN=TBBUF(I)
                U2N=U2BBUF(I)
                U3N=U3BBUF(I)
                ENDIF
                ELSE
                        IF(UCB.LE.0.AND.UCF.GE.0) THEN
                        TN=TNCON(I,J,K)+TNCON(I-1,J,K)-T1(I,J,K)
                        U2N=U2NCON(I,J,K)+U2NCON(I-1,J,K)-U21(I,J,K)
                        U3N=U3NCON(I,J,K)+U3NCON(I-1,J,K)-U31(I,J,K)
                        ELSE
                        ENDIF 
                        
                ENDIF
      
        ENDIF

     

      P1(I,J,K)=PN
      U11(I,J,K)=UN
      RO1(I,J,K)=RON
      T1(I,J,K)=TN
      U21(I,J,K)=U2N
      U31(I,J,K)=U3N

      ENDDO
!     ------------------------------------------------------------------------------------------------------
!     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ВОСТОЧНОЙ ГРАНИЦЕ
!
!      P1(1,J,K)=P1(N1,J,K)
!      U11(1,J,K)=U11(N1,J,K)
!      RO1(1,J,K)=RO1(N1,J,K)
!      T1(1,J,K)=T1(N1,J,K)
!      U21(1,J,K)=U21(N1,J,K)
!      U31(1,J,K)=U31(N1,J,K)
!     -------------------------------------------------------------------------------------------------------

      ENDDO
      ENDDO
  

!     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ГРАНЯХ DS2, ортогональных оси X2    
      
!     ПРОХОД ПО НАПРАВЛЕНИЮ 2 (Y,fi)
  
      DO K=1,N3-1
      DO I=1,N1-1
      j1=0
      j2=N2
      DO j=j1,j2

      XLE=1+(L-1)*(X1(I+1)-1)! геометрический фактор цилиндричности
      XLW=1+(L-1)*(X1(I)-1)  !  геометрический фактор цилиндричности
      XLS=0.5*(XLE+XLW)      !  геометрический фактор цилиндричности
 
      DX=X2(J+1)-X2(J)
 
      UF=U22(I,J+1,K)
      UB=U22(I,J,K)
      UCN=U2NCON(I,J,K)
      UC=U2CON(I,J,K)
       
      U1F=U12(I,J+1,K)
      U1B=U12(I,J,K)
      U1CN=U1NCON(I,J,K)
      U1C=U1CON(I,J,K)
       
      U3F=U32(I,J+1,K)
      U3B=U32(I,J,K)
      U3CN=U3NCON(I,J,K)
      U3C=U3CON(I,J,K)
      
      ROF=RO2(I,J+1,K)
      ROB=RO2(I,J,K)
      ROCN=RONCON(I,J,K)
      ROC=ROCON(I,J,K)

      TF=T2(I,J+1,K)
      TB=T2(I,J,K)
      TCN=TNCON(I,J,K)
      TC=TCON(I,J,K)
       
     
      PF=P2(I,J+1,K)
      PB=P2(I,J,K)
      PCN=SOUND**2*(ROCN-RO0G)
      PC=SOUND**2*(ROC-RO0G)

!     Вычисление инвариантов

      RF=UF+PF/(RO0G*SOUND)
      RB=UB+PB/(RO0G*SOUND)
      RCN=UCN+PCN/(RO0G*SOUND)
      RC=UC+PC/(RO0G*SOUND)
       
      RFN=2*RCN-RB
      
      QF=UF-PF/(RO0G*SOUND)
      QB=UB-PB/(RO0G*SOUND)
      QCN=UCN-PCN/(RO0G*SOUND)
      QC=UC-PC/(RO0G*SOUND)
       
      QBN=2*QCN-QF

      TFN=2*TCN-TB
      TBN=2*TCN-TF

      U1FN=2*U1CN-U1B
      U1BN=2*U1CN-U1F

      U3FN=2*U3CN-U3B
      U3BN=2*U3CN-U3F

!     Допустимый диапазон их изменения

      GR=2*(RCN-RC)/DT+(UCN+SOUND)*(RF-RB)/DX/XLS
      GQ=2*(QCN-QC)/DT+(UCN-SOUND)*(QF-QB)/DX/XLS
      GT=2*(TCN-TC)/DT+UCN*(TF-TB)/DX/XLS
      GU1=2*(U1CN-U1C)/DT+UCN*(U1F-U1B)/DX/XLS
      GU3=2*(U3CN-U3C)/DT+UCN*(U3F-U3B)/DX/XLS
      
      RMAX=DMAX1(RF,RC,RB) +DT*GR
      RMIN=DMIN1(RF,RC,RB) +DT*GR

      QMAX=DMAX1(QF,QC,QB) +DT*GQ
      QMIN=DMIN1(QF,QC,QB) +DT*GQ

      TMAX=DMAX1(TF,TC,TB) +DT*GT
      TMIN=DMIN1(TF,TC,TB) +DT*GT

      U1MAX=DMAX1(U1F,U1C,U1B) +DT*GU1
      U1MIN=DMIN1(U1F,U1C,U1B) +DT*GU1

      U3MAX=DMAX1(U3F,U3C,U3B) +DT*GU3
      U3MIN=DMIN1(U3F,U3C,U3B) +DT*GU3

      

!     Коррекция инвариантов

      IF(RFN.GT.RMAX) RFN=RMAX
      IF(RFN.LT.RMIN) RFN=RMIN

      IF(QBN.GT.QMAX) QBN=QMAX
      IF(QBN.LT.QMIN) QBN=QMIN

      IF(TFN.GT.TMAX) TFN=TMAX
      IF(TFN.LT.TMIN) TFN=TMIN

      IF(TBN.GT.TMAX) TBN=TMAX
      IF(TBN.LT.TMIN) TBN=TMIN

      IF(U1FN.GT.U1MAX) U1FN=U1MAX
      IF(U1FN.LT.U1MIN) U1FN=U1MIN

      IF(U1BN.GT.U1MAX) U1BN=U1MAX
      IF(U1BN.LT.U1MIN) U1BN=U1MIN

      IF(U3FN.GT.U3MAX) U3FN=U3MAX
      IF(U3FN.LT.U3MIN) U3FN=U3MIN

      IF(U3BN.GT.U3MAX) U3BN=U3MAX
      IF(U3BN.LT.U3MIN) U3BN=U3MIN

!     Засылка инвариантов в буферные массивы

      RBUF(J+1)=RFN
      QBUF(J)=QBN
      TFBUF(J+1) =TFN
      TBBUF(J)=TBN
      U2FBUF(J+1)=U1FN
      U2BBUF(J)=U1BN
      U3FBUF(J+1)=U3FN
      U3BBUF(J)=U3BN

      ENDDO

!     ГРАНИЧНЫЕ УСЛОВИЯ ПО НАПРАВЛЕНИЮ 2 (Y,fi)   
!     ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ

!      RBUF(1)=RBUF(N2)      !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TFBUF(1)=TFBUF(N2)
!      U2FBUF(1)=U2FBUF(N2)
!      U3FBUF(1)= U3FBUF(N2)

!      QBUF(N2)=QBUF(1)       !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TBBUF(N2)= TBBUF(1)
!      U2BBUF(N2)=U2BBUF(1)
!      U3BBUF(N2)=U3BBUF(1)

!     Вычисление потоковых переменных  
 
      DO J=1,N2

      RO0B=RONCON(I,J-1,K)
      RO0F=RONCON(I,J,K)

      RN=RBUF(J)
      QN=QBUF(J)
      
      PN=(RN-QN)*SOUND*RO0G/2
      UN=(RN+QN)/2
      
      RON=(RO0G+PN/SOUND**2)
      
      UCF=U2NCON(I,J,K)
      UCB=U2NCON(I,J-1,K)
      
      IF(UCF.GE.0.AND.UCB.GT.0) THEN 
      TN=TFBUF(J)
      U1N=U2FBUF(J)
      U3N=U3FBUF(J) 
      ELSE
        IF(UCF.LE.0.AND.UCB.LE.0) THEN   
        TN=TBBUF(J)
        U1N=U2BBUF(J)
        U3N=U3BBUF(J)
        ELSE
        ENDIF
                IF(UCB.GE.0.AND.UCF.LE.0) THEN
                IF(UCB.GT.-UCF) THEN
                TN=TFBUF(J) 
                U1N=U2FBUF(J)
                U3N=U3FBUF(J)  
                ELSE
                TN=TBBUF(J)
                U1N=U2BBUF(J)
                U3N=U3BBUF(J)
                ENDIF
                ELSE
                        IF(UCB.LE.0.AND.UCF.GE.0) THEN
                        TN=TNCON(I,J,K)+TNCON(I,J-1,K)-T2(I,J,K)
                        U1N=U1NCON(I,J,K)+U1NCON(I,J-1,K)-U12(I,J,K)
                        U3N=U3NCON(I,J,K)+U3NCON(I,J-1,K)-U32(I,J,K)
                        ELSE
                        ENDIF 
                        
                ENDIF
      
        ENDIF

      P2(I,J,K)=PN
      U22(I,J,K)=UN
      RO2(I,J,K)=RON
      T2(I,J,K)=TN
      U12(I,J,K)=U1N
      U32(I,J,K)=U3N

      ENDDO

!     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ЮЖНОЙ ГРАНИЦЕ

!      P2(I,1,K)=P2(I,N2,K)
!      U22(I,1,K)=U22(I,N2,K)
!      RO2(I,1,K)=RO2(I,N2,K)
!      T2(I,1,K)=T2(I,N2,K)
!      U12(I,1,K)=U12(I,N2,K)
!      U32(I,1,K)=U32(I,N2,K)

      ENDDO
      ENDDO
   
!     ВЫЧИСЛЕНИЕ ПОТОКОВЫХ ПЕРЕМЕННЫХ НА ГРАНЯХ DS3, ортогональных оси X3

!     ПРОХОД ПО НАПРАВЛЕНИЮ 3 (Z,Z)

      DO J=1,N2-1
      DO I=1,N1-1
	  k1=0
      k2=N3
      if(coords(2).eq.0)k1=1
      if(coords(2).eq.pz-1)k2=N3-1  
      DO K=k1,k2
     
      DX=X3(K+1)-X3(K)
 
      UF=U33(I,J,K+1)
      UB=U33(I,J,K)
      UCN=U3NCON(I,J,K)
      UC=U3CON(I,J,K)
       
      U2F=U23(I,J,K+1)
      U2B=U23(I,J,K)
      U2CN=U2NCON(I,J,K)
      U2C=U2CON(I,J,K)
       
      U1F=U13(I,J,K+1)
      U1B=U13(I,J,K)
      U1CN=U1NCON(I,J,K)
      U1C=U1CON(I,J,K)
      
!      RO0=RO0G  !!DNINT(RONCON(I,J,K))
       
      ROF=RO3(I,J,K+1)
      ROB=RO3(I,J,K)
      ROCN=RONCON(I,J,K)
      ROC=ROCON(I,J,K)

      TF=T3(I,J,K+1)
      TB=T3(I,J,K)
      TCN=TNCON(I,J,K)
      TC=TCON(I,J,K)
       
     
      PF=P3(I,J,K+1)
      PB=P3(I,J,K)
      PCN=SOUND**2*(ROCN-RO0G)
      PC=SOUND**2*(ROC-RO0G)

!     Вычисление инвариантов

      RF=UF+PF/(RO0G*SOUND)
      RB=UB+PB/(RO0G*SOUND)
      RCN=UCN+PCN/(RO0G*SOUND)
      RC=UC+PC/(RO0G*SOUND)
     
      RFN=2*RCN-RB
      

      QF=UF-PF/(RO0G*SOUND)
      QB=UB-PB/(RO0G*SOUND)
      QCN=UCN-PCN/(RO0G*SOUND)
      QC=UC-PC/(RO0G*SOUND)
        
      QBN=2*QCN-QF

      TFN=2*TCN-TB
      TBN=2*TCN-TF

      U2FN=2*U2CN-U2B
      U2BN=2*U2CN-U2F

      U1FN=2*U1CN-U1B
      U1BN=2*U1CN-U1F

!     Допустимый диапазон их изменения


      GR=2*(RCN-RC)/DT+(UCN+SOUND)*(RF-RB)/DX
      GQ=2*(QCN-QC)/DT+(UCN-SOUND)*(QF-QB)/DX
      GT=2*(TCN-TC)/DT+UCN*(TF-TB)/DX
      GU2=2*(U2CN-U2C)/DT+UCN*(U2F-U2B)/DX
      GU1=2*(U1CN-U1C)/DT+UCN*(U1F-U1B)/DX

      RMAX=DMAX1(RF,RC,RB)  +DT*GR
      RMIN=DMIN1(RF,RC,RB)  +DT*GR

      QMAX=DMAX1(QF,QC,QB)  +DT*GQ
      QMIN=DMIN1(QF,QC,QB)  +DT*GQ

      TMAX=DMAX1(TF,TC,TB) +DT*GT
      TMIN=DMIN1(TF,TC,TB) +DT*GT

      U2MAX=DMAX1(U2F,U2C,U2B) +DT*GU2
      U2MIN=DMIN1(U2F,U2C,U2B) +DT*GU2

      U1MAX=DMAX1(U1F,U1C,U1B) +DT*GU1
      U1MIN=DMIN1(U1F,U1C,U1B) +DT*GU1

!     Коррекция инвариантов

      IF(RFN.GT.RMAX) RFN=RMAX
      IF(RFN.LT.RMIN) RFN=RMIN

      IF(QBN.GT.QMAX) QBN=QMAX
      IF(QBN.LT.QMIN) QBN=QMIN

      IF(TFN.GT.TMAX) TFN=TMAX
      IF(TFN.LT.TMIN) TFN=TMIN

      IF(TBN.GT.TMAX) TBN=TMAX
      IF(TBN.LT.TMIN) TBN=TMIN

      IF(U2FN.GT.U2MAX) U2FN=U2MAX
      IF(U2FN.LT.U2MIN) U2FN=U2MIN

      IF(U2BN.GT.U2MAX) U2BN=U2MAX
      IF(U2BN.LT.U2MIN) U2BN=U2MIN

      IF(U1FN.GT.U1MAX) U1FN=U1MAX
      IF(U1FN.LT.U1MIN) U1FN=U1MIN

      IF(U1BN.GT.U1MAX) U1BN=U1MAX
      IF(U1BN.LT.U1MIN) U1BN=U1MIN

!     Засылка инвариантов в буферные массивы

      RBUF(K+1)=RFN
      QBUF(K)=QBN
      TFBUF(K+1) =TFN
      TBBUF(K)=TBN
      U2FBUF(K+1)=U2FN
      U2BBUF(K)=U2BN
      U3FBUF(K+1)=U1FN
      U3BBUF(K)=U1BN

      ENDDO
 
!     ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ
!     ГРАНИЧНЫЕ УСЛОВИЯ ПО НАПРАВЛЕНИЮ 3 (Z,Z)  

         
!     ЗАДАНИЕ ГРАНИЧНЫХ УСЛОВИЙ ПЕРИОДИЧНОСТИ

!      RBUF(1)=RBUF(N3)      !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TFBUF(1)=TFBUF(N3)
!      U2FBUF(1)=U2FBUF(N3)
!      U3FBUF(1)= U3FBUF(N3)

!      QBUF(N3)=QBUF(1)       !   УСЛОВИЕ ПЕРИОДИЧНОСТИ
!      TBBUF(N3)= TBBUF(1)
!      U2BBUF(N3)=U2BBUF(1)
!      U3BBUF(N3)=U3BBUF(1) 
      
  
!     Вычисление потоковых переменных на внутренних гранях 
      k1=1
      k2=N3
      if(coords(2).eq.0)k1=2
	  if(coords(2).eq.pz-1)k2=N3-1
      DO k=k1,k2
      UCB=U3NCON(I,J,K-1)
      UCF=U3NCON(I,J,K)

      RO0B=RONCON(I,J,K-1)
      RO0F=RONCON(I,J,K)

      RN=RBUF(K)
      QN=QBUF(K)

      PN=(RN-QN)*SOUND*RO0G/2
      UN=(RN+QN)/2
      
 
      
      RON=(RO0G+PN/SOUND**2)
      
      UCF=U3NCON(I,J,K)
      UCB=U3NCON(I,J,K-1)
      
      IF(UCF.GE.0.AND.UCB.GT.0) THEN 
      TN=TFBUF(K)
      U2N=U2FBUF(K)
      U3N=U3FBUF(K) 
      ELSE
        IF(UCF.LE.0.AND.UCB.LE.0) THEN   
        TN=TBBUF(K)
        U2N=U2BBUF(K)
        U1N=U3BBUF(K)
        ELSE
        ENDIF
                IF(UCB.GE.0.AND.UCF.LE.0) THEN
                IF(UCB.GT.-UCF) THEN
                TN=TFBUF(K) 
                U2N=U2FBUF(K)
                U1N=U3FBUF(K)  
                ELSE
                TN=TBBUF(K)
                U2N=U2BBUF(K)
                U1N=U3BBUF(K)
                ENDIF
                ELSE
                        IF(UCB.LE.0.AND.UCF.GE.0) THEN
                        TN=TNCON(I,J,K)+TNCON(I,J,K-1)-T3(I,J,K)
                        U2N=U2NCON(I,J,K)+U2NCON(I,J,K-1)-U23(I,J,K)
                        U1N=U1NCON(I,J,K)+U1NCON(I,J,K-1)-U13(I,J,K)
                        ELSE
                        ENDIF 
                        
                ENDIF
      
        ENDIF

      P3(I,J,K)=PN
      U33(I,J,K)=UN
      RO3(I,J,K)=RON
      T3(I,J,K)=TN
      U23(I,J,K)=U2N
      U13(I,J,K)=U1N

      ENDDO

      !Periodic
!      P3(I,J,1)=P3(I,J,N3)
!      U33(I,J,1)=U33(I,J,N3)
!      RO3(I,J,1)=RO3(I,J,N3)
!      T3(I,J,1)= T3(I,J,N3)
!      U23(I,J,1)=U23(I,J,N3)
!      U13(I,J,1)=U13(I,J,N3)
      if(coords(2).eq.0)then
      !Inlet
      K=1
      QN=QBUF(K)
      RN=U3INLET+(RONCON(I,J,K)-RO0G)*SOUND/RO0G
      UN=(RN+QN)/2
      PN=(RN-QN)*SOUND*RO0G/2
      RON=RO0G+PN/SOUND/SOUND
      U2N=U2INLET
      U1N=U1INLET
      TN=TINLET
      P3(I,J,K)=PN
      U33(I,J,K)=UN
      RO3(I,J,K)=RON
      T3(I,J,K)=TN
      U23(I,J,K)=U2N
      U13(I,J,K)=U1N
      endif
      if(coords(2).eq.pz-1)then
      !Outlet
      K=N3
      RN=RBUF(K)
      PN=POUTLET
      UN=RN-PN/RO0G/SOUND
      TN=TFBUF(K)
      U2N=U2FBUF(K)
      U1N=U3FBUF(K)
      P3(I,J,K)=PN
      U33(I,J,K)=UN
      RO3(I,J,K)=RON
      T3(I,J,K)=TN
      U23(I,J,K)=U2N
      U13(I,J,K)=U1N
      endif
      ENDDO
      ENDDO
   
      END SUBROUTINE PHASE2	  
	  
      end 

