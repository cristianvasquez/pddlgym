(define (problem blocks)
    (:domain conditionalblocks)
    (:objects 
        D - block
        B - block
        A - block
        C - block
        E - block
        robot - robot
    )
    (:init 
        (clear C) 
        (clear A) 
        (clear B) 
        (clear D) 
        (ontable C) 
        (ontable A)
        (ontable B) 
        (ontable E)
        (on D E)
        (handempty robot)

        ; Action literals
        (PickUp A)
        (PutDown A)
        (Stack A B)
        (Stack A C)
        (Stack A D)
        (Stack A E)
        (PickUp B)
        (PutDown B)
        (Stack B A)
        (Stack B C)
        (Stack B D)
        (Stack B E)
        (PickUp C)
        (PutDown C)
        (Stack C B)
        (Stack C A)
        (Stack C D)
        (Stack C E)
        (PickUp D)
        (PutDown D)
        (Stack D B)
        (Stack D C)
        (Stack D A)
        (Stack D E)
        (PickUp E)
        (PutDown E)
        (Stack E B)
        (Stack E C)
        (Stack E A)
        (Stack E D)

    )
    (:goal (and (on A B) (on B C) (on E D) (clear E) (clear A)))
)
