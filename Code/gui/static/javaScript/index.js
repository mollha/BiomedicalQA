document.addEventListener("DOMContentLoaded", function(){
	let increments = [0, 0, 0]

	let question_input = $('#question-input');
	let context_input = $('#context-input');
	let question_type_input = $('#question-type-input');
	let predicted_answer_input = $('#answer-input');
	let expected_answer_input = $('#expected-answer-input');

	const yesno_examples = [
		["Can Freund's complete adjuvant induce arthritis?", "Rheumatoid arthritis (RA) was induced by Freunds Complete Adjuvant (FCA; 1 mg/0.1 ml paraffin oil), injected subcutaneously on days 0, 30 and 40", ["yes"]]
	];

	const factoid_examples = [
		["Which cells produce Interleukin 17A?", "Several studies have shown an increased expression/release of Th17 related cytokine, IL-17A in ASD.", ["TH17"]]
	];

	const list_examples = [
		["List two medication included in the Juluca pill.", "Dolutegravir/rilpivirine (Juluca\u00ae) is the first two-drug single-tablet regimen (STR) to be approved for the treatment of HIV-1 infection in adults.", ["dolutegravir" , "rilpivirine"]],
		["Which drugs are included in the EE-4A regimen for Wilm\u0027s tumor?", "Five patients received treatment regimen EE-4A, dactinomycin, and vincristine.", ["dactinomycin", "vincristine"]]
	];

	function get_next_example(examples, q_type){
		console.log(examples);
		let idx_of_increment = ['yesno', 'factoid', 'list'].indexOf(q_type);
		console.log(idx_of_increment);
		console.log(increments[idx_of_increment]);
		increments[idx_of_increment] = (increments[idx_of_increment] + 1) % examples.length;
		console.log(increments[idx_of_increment]);
		return examples[increments[idx_of_increment]]
	}

	function populate_example(q_type){
		let example = ["", "", ""]
		if(q_type === "yesno"){
			example = get_next_example(yesno_examples, q_type);
		} else if(q_type === "factoid"){
			example = get_next_example(factoid_examples, q_type);
		} else if(q_type === "list"){ // list
			 example = get_next_example(list_examples, q_type);
		} else {
			// resetting
			document.getElementById("first-opt").selected = true;
			question_input.val("");
			context_input.val("");
			expected_answer_input.val("");
			predicted_answer_input.val("");
			return
		}

		console.log('Example', example);

		question_type_input.val(q_type);
		const question = example[0];
		question_input.val(question);

		const context = example[1];
		context_input.val(context);

		const answer = example[2];
		let combined_answer = "";
		if(answer.length > 1){
			for(let i=0; i < answer.length; i++) {
				let cap_a = answer[i].charAt(0).toUpperCase() + answer[i].slice(1);
				combined_answer += (i + 1).toString() + ". " + cap_a + "\n";
			}
		} else {
			combined_answer = answer.pop();
		}
		expected_answer_input.val(combined_answer.charAt(0).toUpperCase() + combined_answer.slice(1));
	}

	//buttons
	const reset_button = document.getElementById("reset-button");
	let factoid_button = document.getElementById("factoid-button");
	let yesno_button = document.getElementById("yesno-button");
	let list_button = document.getElementById("list-button");

	reset_button.addEventListener('click', function (){ return populate_example("reset") });
	factoid_button.addEventListener('click', function (){ populate_example("reset"); return populate_example("factoid")});
	yesno_button.addEventListener('click', function (){ populate_example("reset"); return populate_example("yesno")});
	list_button.addEventListener('click', function (){ populate_example("reset"); return populate_example("list")});


	function invalid_input(element){
		element.css("box-shadow", "0 0 10px rgb(255, 0, 0)");
	}

	// When the user ID form is submitted, POST input to /process and if the user ID exists, redirect to welcome page
	$('#form').on('submit', function(event) {
		$('#answer-input').val("");
		$.ajax({
			data: {
			question: question_input.val(),
			context: context_input.val(),
			question_type: question_type_input.val()
			},
			type: 'POST',
			url: '/handlepost',
			success: function (response) {
				if (!response) {
					invalid_input(question_input);
					invalid_input(context_input);
				} else {
					console.log(response)
					console.log(response["prediction"])
					let p_answer = response["prediction"];

					let c_answer = "";
					if(p_answer.length > 1){
						for(let i=0; i < p_answer.length; i++) {
							let cap_a = p_answer[i].charAt(0).toUpperCase() + p_answer[i].slice(1);
							c_answer += (i + 1).toString() + ". " + cap_a + "\n";
						}
						} else {
							c_answer = p_answer.pop();
						}
					$('#answer-input').val(c_answer.charAt(0).toUpperCase() + c_answer.slice(1));
				}
				return response;
			}
		});
		// HTML automatically tries to post the form, we therefore manually stop this
		event.preventDefault();
	});

});