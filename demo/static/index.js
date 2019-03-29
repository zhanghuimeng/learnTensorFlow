function loadTestData(index) {
    var table = $("#test-data-table")[0];
    var cell = table.rows[index].cells[1]; // This is a DOM "TD" element
    var $cell = $(cell); // Now it's a jQuery object.
    console.log("index = " + index);
    console.log("Cell text = " + cell.text());
    $("#src").val(cell.text());
}
