# devIE 
> devIE is Inference Engineering

- Ideas
    1. organic operation with memory: transformer continuous inference with occation external/user input. Past token as memory. Sleep phase: drop past token by probabilty proportional to attention weight accumulation. Remember by reconstructing past sequence with remaning tokens.... more extention => clustering of tape for extra connection... [propagational attention] only attend to nearby things first, and consequtively include more tokens if not ideal.
    2. mid sequence regen: plan (code comment) and then implement step generation (actual coding) inside plan sequence
