(define (problem minecraft) 
    (:domain minecraft)

    (:objects
    
	log-0 - moveable
	grass-1 - moveable
	grass-2 - moveable
	grass-3 - moveable
	log-4 - moveable
	grass-5 - moveable
	new-0 - moveable
	new-1 - moveable
	new-2 - moveable
	agent - agent
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
    )

    (:init
    
	(Hypothetical new-0)
	(Hypothetical new-1)
	(Hypothetical new-2)
	(IsLog log-0)
	(IsGrass grass-1)
	(IsGrass grass-2)
	(IsGrass grass-3)
	(IsLog log-4)
	(IsGrass grass-5)
	(At log-0 loc-0-1)
	(At grass-1 loc-1-0)
	(At grass-2 loc-0-3)
	(At grass-3 loc-0-1)
	(At log-4 loc-0-1)
	(At grass-5 loc-0-0)
	(AgentAt loc-2-3)
	(Handsfree agent)

    ; Action literals
    
	(recall log-0)
	(craftplank log-0 grass-1)
	(craftplank log-0 grass-2)
	(craftplank log-0 grass-3)
	(craftplank log-0 log-4)
	(craftplank log-0 grass-5)
	(craftplank log-0 new-0)
	(craftplank log-0 new-1)
	(craftplank log-0 new-2)
	(equip log-0)
	(pick log-0)
	(recall grass-1)
	(craftplank grass-1 log-0)
	(craftplank grass-1 grass-2)
	(craftplank grass-1 grass-3)
	(craftplank grass-1 log-4)
	(craftplank grass-1 grass-5)
	(craftplank grass-1 new-0)
	(craftplank grass-1 new-1)
	(craftplank grass-1 new-2)
	(equip grass-1)
	(pick grass-1)
	(recall grass-2)
	(craftplank grass-2 log-0)
	(craftplank grass-2 grass-1)
	(craftplank grass-2 grass-3)
	(craftplank grass-2 log-4)
	(craftplank grass-2 grass-5)
	(craftplank grass-2 new-0)
	(craftplank grass-2 new-1)
	(craftplank grass-2 new-2)
	(equip grass-2)
	(pick grass-2)
	(recall grass-3)
	(craftplank grass-3 log-0)
	(craftplank grass-3 grass-1)
	(craftplank grass-3 grass-2)
	(craftplank grass-3 log-4)
	(craftplank grass-3 grass-5)
	(craftplank grass-3 new-0)
	(craftplank grass-3 new-1)
	(craftplank grass-3 new-2)
	(equip grass-3)
	(pick grass-3)
	(recall log-4)
	(craftplank log-4 log-0)
	(craftplank log-4 grass-1)
	(craftplank log-4 grass-2)
	(craftplank log-4 grass-3)
	(craftplank log-4 grass-5)
	(craftplank log-4 new-0)
	(craftplank log-4 new-1)
	(craftplank log-4 new-2)
	(equip log-4)
	(pick log-4)
	(recall grass-5)
	(craftplank grass-5 log-0)
	(craftplank grass-5 grass-1)
	(craftplank grass-5 grass-2)
	(craftplank grass-5 grass-3)
	(craftplank grass-5 log-4)
	(craftplank grass-5 new-0)
	(craftplank grass-5 new-1)
	(craftplank grass-5 new-2)
	(equip grass-5)
	(pick grass-5)
	(recall new-0)
	(craftplank new-0 log-0)
	(craftplank new-0 grass-1)
	(craftplank new-0 grass-2)
	(craftplank new-0 grass-3)
	(craftplank new-0 log-4)
	(craftplank new-0 grass-5)
	(craftplank new-0 new-1)
	(craftplank new-0 new-2)
	(equip new-0)
	(pick new-0)
	(recall new-1)
	(craftplank new-1 log-0)
	(craftplank new-1 grass-1)
	(craftplank new-1 grass-2)
	(craftplank new-1 grass-3)
	(craftplank new-1 log-4)
	(craftplank new-1 grass-5)
	(craftplank new-1 new-0)
	(craftplank new-1 new-2)
	(equip new-1)
	(pick new-1)
	(recall new-2)
	(craftplank new-2 log-0)
	(craftplank new-2 grass-1)
	(craftplank new-2 grass-2)
	(craftplank new-2 grass-3)
	(craftplank new-2 log-4)
	(craftplank new-2 grass-5)
	(craftplank new-2 new-0)
	(craftplank new-2 new-1)
	(equip new-2)
	(pick new-2)
	(move loc-0-0)
	(move loc-0-1)
	(move loc-0-2)
	(move loc-0-3)
	(move loc-1-0)
	(move loc-1-1)
	(move loc-1-2)
	(move loc-1-3)
	(move loc-2-0)
	(move loc-2-1)
	(move loc-2-2)
	(move loc-2-3)
    )

    (:goal (and  (Equipped new-0 agent)  (Agentat loc-2-0) ))
)
    