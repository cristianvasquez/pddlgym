(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	bear-0
	bear-1
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-1-0
	loc-1-1
	loc-1-2
	loc-2-0
	loc-2-1
	loc-2-2
    )

    (:init
    
	(IsBear bear-0)
	(IsBear bear-1)
	(IsRobot robot)
	(At bear-0 loc-0-2)
	(At bear-1 loc-0-2)
	(At robot loc-1-2)
	(Handsfree robot)

    ; Action literals
    
	(Pick bear-0)
	(Place bear-0)
	(Pick bear-1)
	(Place bear-1)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
    )

    (:goal (and  (Holding bear-1)  (At bear-1 loc-2-2) ))
)
    