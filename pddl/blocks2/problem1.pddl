(define (problem blocks)
    (:domain blocks)
    (:objects 
        A - block
        B - block
        robot - robot
    )
    (:init 
        (clear A) 
        (clear B) 
        (ontable A)
        (ontable B) 
        (handempty robot)

        ; Action literals
        (PickUp A)
        (PutDown A)
        (Unstack A)
        (Stack A B)
        (PickUp B)
        (PutDown B)
        (Unstack B)
        (Stack B A)

    )
    (:goal (and (on A B)))
)