(define (problem rearrangement-notyping) 
    (:domain rearrangement-notyping)

    (:objects
    
	pawn-0
	monkey-1
	bear-2
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
	loc-3-0
	loc-3-1
	loc-3-2
    )

    (:init
    
	(IsPawn pawn-0)
	(IsMonkey monkey-1)
	(IsBear bear-2)
	(IsRobot robot)
	(At pawn-0 loc-3-0)
	(At monkey-1 loc-1-2)
	(At bear-2 loc-0-2)
	(At robot loc-0-1)
	(Handsfree robot)

    ; Action literals
    
	(Pick pawn-0)
	(Place pawn-0)
	(Pick monkey-1)
	(Place monkey-1)
	(Pick bear-2)
	(Place bear-2)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
    )

    (:goal (and  (At pawn-0 loc-3-2) ))
)
    