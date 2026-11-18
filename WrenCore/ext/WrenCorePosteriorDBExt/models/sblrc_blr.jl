# PosteriorDB: sblrc-blr (the blr Stan model with the sblrc dataset)
module SblrcBLR

using PosteriorDB
using ..SblriBLR: sblri_blr

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "sblrc-blr")))
end

const wren_model = sblri_blr(data["y"], data["X"], data["D"])

end
