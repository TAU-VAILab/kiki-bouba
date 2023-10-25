var s_elt = document.getElementById('gall-select');
var i_elt = document.getElementById('gall-img');

function gallery_change() {
    var value = s_elt.value;
    if (value === "none") {
        i_elt.innerHTML = "";
    } else {
        var filename = "assets/gallery_imgs/" + value + ".jpg";
        i_elt.innerHTML = '<img src="' + filename + '"></img>';
    }
}

s_elt.onchange = gallery_change;