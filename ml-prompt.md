This is great. Okay, let's talk about the scoring criteria and mechanism. I think your insight was an excellent one when you added the price to rent ratio, that's very helpful. In a sense, that's 
valuable -- how much rent can I get for this property. I suppose another criteria for deciding if it's worth investing is if I want to "flip" the house -- if the house is expected to rise rapidly in price,
 maybe in 5-10 years I could sell it and double my money? 

In designing the algorithm right now, I've injected my inductive bias, which is that a growing south asian population means that there's an influx of educated tech workers who will drive prices up. That's 
an imperfect assumption.

Putting these 2 thoughts together, I'm wondering if we can design an objective function that aligns with the 2 possible things we want to optimize for (low price-to-rent ratio today and booming prices 
later) and that learns weights on the parameters we have available to us now. Even if we forget about the "learning" bit, I'm at least wondering if there's any way we can back-test our current algorithm on
 historical data to see how well it would have predicted the "winning" neighborhoods/areas of years past.