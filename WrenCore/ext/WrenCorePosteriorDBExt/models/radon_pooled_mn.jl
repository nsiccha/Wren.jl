# PosteriorDB: radon_mn-radon_pooled
module RadonPooledMN

using PosteriorDB
using ..RadonPooledAll: radon_pooled_model

const data = let
    pdb = PosteriorDB.database()
    PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, "radon_mn-radon_pooled")))
end

const wren_model = radon_pooled_model(data["log_radon"], data["floor_measure"])

end
