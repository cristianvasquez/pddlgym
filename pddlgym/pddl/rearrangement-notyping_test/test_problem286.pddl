(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	monkey-0
	bear-1
	pawn-2
	robot
	loc-0-0
	loc-0-1
	loc-0-2
	loc-0-3
	loc-1-0
	loc-1-1
	loc-1-2
	loc-1-3
	loc-2-0
	loc-2-1
	loc-2-2
	loc-2-3
    )

    (:init
    
	(IsMonkey monkey-0)
	(IsBear bear-1)
	(IsPawn pawn-2)
	(IsRobot robot)
	(At monkey-0 loc-1-3)
	(At bear-1 loc-2-2)
	(At pawn-2 loc-2-0)
	(At robot loc-0-1)
	(Handsfree robot)

    ; Action literals
    
	(Pick monkey-0)
	(Place monkey-0)
	(Pick bear-1)
	(Place bear-1)
	(Pick pawn-2)
	(Place pawn-2)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-0-3)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-1-3)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-2-3)
    )

    (:goal (and  (At monkey-0 loc-2-0) ))
)
    