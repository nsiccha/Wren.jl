# PosteriorDB: three_men{1,2,3}-ldaK2
module LDAThreeMen

using PosteriorDB
using ..LDAThreeDocs: lda_k2

function load_case(name, M)
    pdb = PosteriorDB.database()
    data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
    data["V"] == 249 && data["M"] == M && data["N"] == 4999 ||
        error("unexpected public Three Men LDA fixture: $name")
    return lda_k2(data["w"], data["doc"], data["V"], data["M"])
end

const case1 = (wren_model=load_case("three_men1-ldaK2", 6),)
const case2 = (wren_model=load_case("three_men2-ldaK2", 14),)
const case3 = (wren_model=load_case("three_men3-ldaK2", 9),)

end
