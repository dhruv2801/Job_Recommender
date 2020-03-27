$(document).ready(
    function(){
        $('button:submit').attr('disabled',true);
        $('input:file').change(
            function(){
                if ($(this).val()){
                    $('button:submit').removeAttr('disabled'); 
                }
                else {
                    $('button:submit').attr('disabled',true);
                }
            });
    });