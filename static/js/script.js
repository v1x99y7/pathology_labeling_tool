var pop = document.getElementById("pop");

function show_buttons(element, label_id){
    var img_id = element.id;
    element.style.border = "4px solid yellow";

    pop.innerHTML = ``
    pop.style.border = "1px solid white";
    pop.style.backgroundColor = "white";

    setTimeout(function(){
        pop.innerHTML = `
        <div>
            <button class="pop-button" onclick="select_button('ADI', '${img_id}', '${label_id}')">ADI</button>
            <button class="pop-button" onclick="select_button('BACK', '${img_id}', '${label_id}')">BACK</button>
            <button class="pop-button" onclick="select_button('DEB', '${img_id}', '${label_id}')">DEB</button>
        </div>
        <div>
            <button class="pop-button" onclick="select_button('LYM', '${img_id}', '${label_id}')">LYM</button>
            <button class="pop-button" onclick="select_button('MUC', '${img_id}', '${label_id}')">MUC</button>
            <button class="pop-button" onclick="select_button('MUS', '${img_id}', '${label_id}')">MUS</button>
        </div>
        <div>
            <button class="pop-button" onclick="select_button('NORM', '${img_id}', '${label_id}')">NORM</button>
            <button class="pop-button" onclick="select_button('STR', '${img_id}', '${label_id}')">STR</button>
            <button class="pop-button" onclick="select_button('TUM', '${img_id}', '${label_id}')">TUM</button>
        </div>
        `;
        pop.style.border = "3px solid red";
        pop.style.backgroundColor = "lightgray";
    }, 100)
}

function select_button(tissue, img_id, label_id){
    var label = document.getElementById(label_id);
    label.value = tissue;

    var img = document.getElementById(img_id);
    img.style.border = "4px solid red";
    
    pop.innerHTML = ``;
    pop.style.border = "1px solid white";
    pop.style.backgroundColor = "white";
}